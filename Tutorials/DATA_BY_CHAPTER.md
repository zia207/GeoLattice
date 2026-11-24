# Tutorial Data by Chapter

The data each tutorial notebook reads, organised by chapter, sub-chapter and notebook. Paths are relative to the repository root. Built from `Tutorials/DATA_INVENTORY.csv` by `Tutorials/_tools/data_by_chapter.py`; the same tables are in `Tutorials/DATA_BY_CHAPTER.xlsx` (one sheet per chapter).

## Contents

- [Summary](#summary)
- [01_vector · Vector Data](#01_vector--vector-data)
- [02_raster · Raster Data](#02_raster--raster-data)
- [03_spatial_analysis · Spatial Analysis](#03_spatial_analysis--spatial-analysis)
- [04_spatial_statistics · Spatial Statistics](#04_spatial_statistics--spatial-statistics)
- [05_geostatistics · Geostatistics](#05_geostatistics--geostatistics)
- [06_spatialDB · Spatial Databases](#06_spatialDB--spatial-databases)
- [07_web_cloud_gis · Web and Cloud GIS](#07_web_cloud_gis--web-and-cloud-gis)
- [08_remote_sensing · Remote Sensing](#08_remote_sensing--remote-sensing)
- [Datasets shared between chapters](#datasets-shared-between-chapters)

## Summary

| Chapter | Title | Notebooks with data | Dataset folders | Data files | Size | Download URLs |
|---|---|---:|---:|---:|---:|---:|
| 01_vector | Vector Data | 6 | 3 | 47 | 66.5 MB | 0 |
| 02_raster | Raster Data | 4 | 2 | 78 | 198.4 MB | 0 |
| 03_spatial_analysis | Spatial Analysis | 38 | 12 | 237 | 2.6 GB | 3 |
| 04_spatial_statistics | Spatial Statistics | 74 | 1 | 18 | 295.8 MB | 0 |
| 05_geostatistics | Geostatistics | 47 | 1 | 88 | 126.5 MB | 0 |
| 06_spatialDB | Spatial Databases | 14 | 10 | 50 | 618.3 MB | 21 |
| 07_web_cloud_gis | Web and Cloud GIS | 12 | 5 | 116 | 1.4 GB | 19 |
| 08_remote_sensing | Remote Sensing | 113 | 72 | 564 | 120.7 GB | 52 |

## 01_vector · Vector Data

**Dataset folders used in this chapter**

| Dataset folder | Notebooks | Files | Size |
|---|---:|---:|---:|
| `Data/00_shared/vector_data` | 6 | 34 | 76.3 MB |
| `Data/00_shared/data_gdal` | 1 | 11 | 159.2 KB |
| `Data/00_shared/tabular_data` | 1 | 2 | 1.1 MB |

**01_00_vector_data.ipynb**

- `Data/00_shared/vector_data/USA/US_COUNTY.shp` (3.1 MB)

**01_01_read_write_vector_data.ipynb**

- `Data/00_shared/data_gdal/poly.shp` (4.5 KB)
- `Data/00_shared/data_gdal/wkt.csv` (209 B)
- `Data/00_shared/data_gdal/geojson/grenada.geojson` (12.6 KB)
- `Data/00_shared/data_gdal/gml/gnis_pop_110.gml` (13.9 KB)
- `Data/00_shared/data_gdal/kml/placemark.kml` (277 B)
- `Data/00_shared/data_gdal/kml/samples.kml` (35.9 KB)
- `Data/00_shared/data_gdal/openfilegdb/polygon_golden.gdb` (50.0 KB)
- `Data/00_shared/data_gdal/osm/test.osm` (4.3 KB)
- `Data/00_shared/data_gdal/osm/two_points.pbf` (275 B)
- `Data/00_shared/data_gdal/sqlite/poly_spatialite.sqlite` (31.0 KB)
- `Data/00_shared/data_gdal/testfgb/poly.fgb` (6.4 KB)
- `Data/00_shared/vector_data/USA/NY_County_PROJ.shp` (151.2 KB)
- `Data/00_shared/vector_data/output_folder/gdal_as_shapefile/`: 7 files (.shp), 14.3 KB *(written by a notebook)*
- `Data/00_shared/vector_data/output_folder/ny_county/`: 9 files (.csv, .fgb, .gdb, .geojson, .gml, .gpkg, .kml, .shp, .sqlite), 9.3 MB *(written by a notebook)*

**01_02_map_projection_vector_data.ipynb**

- `Data/00_shared/vector_data/Bangladesh/Shapefiles/bd_district.shp` (16.2 MB)
- `Data/00_shared/vector_data/USA/NY_County_GCS.shp` (151.2 KB)
- `Data/00_shared/vector_data/USA/NY_County_PROJ.shp` (151.2 KB)
- `Data/00_shared/vector_data/output_folder/bangladesh/bd_district_BTM.shp` (16.2 MB) *(written by a notebook)*
- `Data/00_shared/vector_data/output_folder/bangladesh/bd_district_GCS.shp` (16.2 MB) *(written by a notebook)*
- `Data/00_shared/vector_data/output_folder/ny_county/NY_County_PROJ.shp` (151.2 KB) *(written by a notebook)*

**01_03_visualizing_vector_data.ipynb**

- `Data/00_shared/tabular_data/bgs_geochemical.csv` (795.7 KB)
- `Data/00_shared/tabular_data/usa_diabetes_data.csv` (334.8 KB)
- `Data/00_shared/vector_data/Bangladesh/Shapefiles/BD_Banladesh_BUTM.shp` (2.2 MB) *(by file name)*
- `Data/00_shared/vector_data/USA/Ononda_Street_PROJ.shp` (175.5 KB) *(by file name)*
- `Data/00_shared/vector_data/USA/US_COUNTY.shp` (3.1 MB) *(by file name)*
- `Data/00_shared/vector_data/output_folder/bangladesh/bgs_geochemical_butm.shp` (96.7 KB) *(written by a notebook)*
- `Data/00_shared/vector_data/output_folder/bangladesh/bgs_geochemical_gcs.shp` (96.7 KB) *(written by a notebook)*

**01_04_geoprocessing_vector_data.ipynb**

- `Data/00_shared/vector_data/USA/CO_SOC_data.shp` (3.8 KB) *(by file name)*
- `Data/00_shared/vector_data/USA/Ononda_Street_PROJ.shp` (175.5 KB) *(by file name)*
- `Data/00_shared/vector_data/USA/US_COUNTY.shp` (3.1 MB) *(by file name)*
- `Data/00_shared/vector_data/USA/US_STATE.shp` (1.1 MB) *(by file name)*
- `Data/00_shared/vector_data/USA/Yellow_Stone.shp` (89.3 KB) *(by file name)*
- `Data/00_shared/vector_data/USA/GP_STATE/*.shp` (0 B)
- `Data/00_shared/vector_data/USA/GP_STATE/CO_STATE_BD.shp` (6.3 KB) *(by file name)*
- `Data/00_shared/vector_data/USA/GP_STATE/KS_STATE_BD.shp` (15.8 KB) *(by file name)*
- `Data/00_shared/vector_data/USA/GP_STATE/NM_STATE_BD.shp` (6.8 KB) *(by file name)*
- `Data/00_shared/vector_data/USA/GP_STATE/WY_STATE_BD.shp` (5.2 KB) *(by file name)*

**01_05_geometry_operation_vector_data.ipynb**

- `Data/00_shared/vector_data/USA/CO_SOC_data.shp` (3.8 KB) *(by file name)*
- `Data/00_shared/vector_data/USA/Ononda_Street_PROJ.shp` (175.5 KB) *(by file name)*
- `Data/00_shared/vector_data/USA/US_COUNTY.shp` (3.1 MB) *(by file name)*
- `Data/00_shared/vector_data/USA/US_STATE.shp` (1.1 MB) *(by file name)*
- `Data/00_shared/vector_data/USA/Yellow_Stone.shp` (89.3 KB) *(by file name)*

## 02_raster · Raster Data

**Dataset folders used in this chapter**

| Dataset folder | Notebooks | Files | Size |
|---|---:|---:|---:|
| `Data/00_shared/raster_data` | 4 | 76 | 166.8 MB |
| `Data/00_shared/vector_data` | 1 | 2 | 32.5 MB |

**02_00_raster_data.ipynb**

- `Data/00_shared/raster_data/gdal_samples/small_world.tif` (234.9 KB)

**02_01_read_write_raster_data.ipynb**

- `Data/00_shared/raster_data/gdal_samples/`: 7 files (.tif), 515.4 KB
- `Data/00_shared/raster_data/gdal_samples/aaigrid/pixel_per_line.asc` (2.9 KB)
- `Data/00_shared/raster_data/gdal_samples/envi/envi_rgbsmall_bsq.img` (7.2 KB)
- `Data/00_shared/raster_data/gdal_samples/hfa/dem10.img` (14.1 KB)
- `Data/00_shared/raster_data/gdal_samples/jpeg/albania.jpg` (12.3 KB)
- `Data/00_shared/raster_data/gdal_samples/jpeg/rgbsmall_rgb.jpg` (2.4 KB)
- `Data/00_shared/raster_data/gdal_samples/png/test.png` (8.5 KB)
- `Data/00_shared/raster_data/gdal_samples/usgsdem/39079G6_truncated.dem` (3.0 KB)
- `Data/00_shared/raster_data/output_folder/`: 9 files (, .asc, .img, .jpg, .png, .tif), 185.6 KB *(written by a notebook)*

**02_02_map_projection_raster.ipynb**

- `Data/00_shared/raster_data/Bangladesh/raj_DEM_GCS.tif` (307.9 KB)
- `Data/00_shared/raster_data/Bangladesh/BD_DEM/GCS/*.tif` (0 B)
- `Data/00_shared/raster_data/USA/Onondaga_DEM.tif` (655.7 KB)
- `Data/00_shared/raster_data/USA/DEM_TILES/Onondaga_DEM[0-9].TIF` (0 B)
- `Data/00_shared/raster_data/output_folder/bangladesh/raj_DEM_BTM.tif` (3.8 MB) *(written by a notebook)*
- `Data/00_shared/raster_data/output_folder/bangladesh/raj_DEM_BTM_warp.tif` (2.0 MB) *(written by a notebook)*
- `Data/00_shared/raster_data/output_folder/bangladesh/BD_DEM/*_PROJ.tif` (0 B) *(written by a notebook)*
- `Data/00_shared/raster_data/output_folder/usa/NY_DEM0_PROJ.tif` (1.3 MB) *(written by a notebook)*
- `Data/00_shared/raster_data/output_folder/usa/DEM_TILES/*_PROJ.tif` (0 B) *(written by a notebook)*

**02_03_basic_raster_operation.ipynb**

- `Data/00_shared/raster_data/Bangladesh/hilly_DEM_UTM.tif` (3.4 MB)
- `Data/00_shared/raster_data/Bangladesh/hilly_dem_BTM.tiff` (1.5 MB)
- `Data/00_shared/raster_data/Bangladesh/HILLY_DEM/*_dem_BTM.tiff` (0 B)
- `Data/00_shared/raster_data/USA/CO_COUNTY_PROJ.shp` (62.9 KB)
- `Data/00_shared/raster_data/USA/CO_ELEV.tif` (41.6 KB)
- `Data/00_shared/raster_data/USA/NY_DEM0_PROJ.tif` (821.8 KB)
- `Data/00_shared/raster_data/USA/Onondaga_DEM.tif` (655.7 KB)
- `Data/00_shared/raster_data/USA/DEM_TILES/Onondaga_DEM[0-9]_PROJ.tif` (0 B)
- `Data/00_shared/raster_data/output_folder/bangladesh/ops/`: 32 files (.gpkg, .tif, .txt), 149.5 MB *(written by a notebook)*
- `Data/00_shared/raster_data/output_folder/usa/ops/co_elev_agg_x2.tif` (13.4 KB) *(written by a notebook)*
- `Data/00_shared/raster_data/output_folder/usa/ops/co_elev_focal_mean.tif` (51.9 KB) *(written by a notebook)*
- `Data/00_shared/raster_data/output_folder/usa/ops/co_elev_stack.tif` (101.8 KB) *(written by a notebook)*
- `Data/00_shared/raster_data/output_folder/usa/ops/onondaga_clip.tif` (354.4 KB) *(written by a notebook)*
- `Data/00_shared/raster_data/output_folder/usa/ops/onondaga_mosaic.tif` (1.4 MB) *(written by a notebook)*
- `Data/00_shared/vector_data/Bangladesh/bd_district_BTM.shp` (16.2 MB)
- `Data/00_shared/vector_data/Bangladesh/bd_district_rice_production_2018_2019_BTM.shp` (16.2 MB)

## 03_spatial_analysis · Spatial Analysis

**Dataset folders used in this chapter**

| Dataset folder | Notebooks | Files | Size |
|---|---:|---:|---:|
| `Data/00_shared/raster_data` | 8 | 59 | 82.4 MB |
| `Data/03_spatial_analysis/MCSM_data` | 8 | 17 | 30.0 MB |
| `Data/03_spatial_analysis/data_segregation` | 8 | 10 | 20.1 MB |
| `Data/00_shared/vector_data` | 7 | 11 | 172.0 MB |
| `Data/00_shared/data_spatial_net` | 6 | 9 | 2.7 MB |
| `Data/03_spatial_analysis/capstone_spatial_analysis` | 5 | 69 | 242.6 MB |
| `Data/03_spatial_analysis/NetCDF` | 5 | 27 | 1.0 GB |
| `Data/03_spatial_analysis/Great_plain` | 2 | 6 | 1.1 GB |
| `Data/00_shared/tabular_data` | 1 | 1 | 7.8 KB |
| `Data/03_spatial_analysis/Data_and_Code_Forest_Musk_Deer` | 1 | 3 | 82.6 MB |
| `Data/03_spatial_analysis/Wheat_suitability` | 1 | 11 | 5.6 MB |
| `Data/03_spatial_analysis/Sylhet_flood_susceptibility_mapping` | 1 | 14 | 135.9 MB |

### 03_01_foundation

**03_01_02_vector_analyis.ipynb**

- `Data/00_shared/tabular_data/bd_poverty.csv` (7.8 KB)
- `Data/00_shared/vector_data/Bangladesh/bd_district_rice_production_2018_2019_BTM.shp` (16.2 MB) *(by file name)*
- `Data/00_shared/vector_data/Bangladesh/Shapefiles/bd_upzila_BTM.shp` (43.5 MB) *(by file name)*
- `Data/00_shared/vector_data/USA/CO_SOC_data.shp` (3.8 KB) *(by file name)*
- `Data/00_shared/vector_data/USA/NY_County_PROJ.shp` (151.2 KB) *(by file name)*
- `Data/00_shared/vector_data/USA/Ononda_Street_PROJ.shp` (175.5 KB) *(by file name)*
- `Data/00_shared/vector_data/USA/US_COUNTY.shp` (3.1 MB) *(by file name)*
- `Data/00_shared/vector_data/USA/US_STATE.shp` (1.1 MB) *(by file name)*
- `Data/00_shared/vector_data/USA/Yellow_Stone.shp` (89.3 KB) *(by file name)*

**03_01_03_raster_analysis.ipynb**

- `Data/00_shared/raster_data/Bangladesh/hilly_dem_BTM.tiff` (1.5 MB)
- `Data/00_shared/raster_data/output_folder/bangladesh/raster_analysis/`: 10 files (.gpkg, .tif), 2.6 MB *(written by a notebook)*
- `Data/00_shared/vector_data/Bangladesh/bd_district_BTM.shp` (16.2 MB)

### 03_02_digital_terrain_modeling

**03_02_00_digital_terrain_modeling_introduction.ipynb**

- `Data/00_shared/raster_data/Bangladesh/hilly_dem_BTM.tiff` (1.5 MB)
- `Data/00_shared/raster_data/output_folder/bangladesh/dtm_intro/intro_hillshade.tif` (677.2 KB) *(written by a notebook)*

**03_02_01_dem_preprocessing.ipynb**

- `Data/00_shared/raster_data/Bangladesh/hilly_dem_BTM.tiff` (1.5 MB)
- `Data/00_shared/raster_data/output_folder/bangladesh/dem_preprocess/aoi_dem_250m.tif` (150.7 KB) *(written by a notebook)*
- `Data/00_shared/raster_data/output_folder/bangladesh/dem_preprocess/aoi_dem_500m.tif` (38.1 KB) *(written by a notebook)*
- `Data/00_shared/raster_data/output_folder/bangladesh/dem_preprocess/aoi_dem_resampled_250m.tif` (150.7 KB) *(written by a notebook)*
- `Data/00_shared/raster_data/output_folder/bangladesh/dem_preprocess/aoi_hillshade.tif` (38.0 KB) *(written by a notebook)*
- `Data/00_shared/raster_data/output_folder/bangladesh/dem_preprocess/aoi_hillshade_multidirectional.tif` (38.0 KB) *(written by a notebook)*

**03_02_02_terrain_analysis.ipynb**

- `Data/00_shared/raster_data/Bangladesh/hilly_dem_BTM.tiff` (1.5 MB)
- `Data/00_shared/raster_data/output_folder/bangladesh/terrain/`: 18 files (, .gpkg, .tif), 2.0 MB *(written by a notebook)*

**03_02_03_hydrology_modeling.ipynb**

- `Data/00_shared/raster_data/Bangladesh/hilly_dem_BTM.tiff` (1.5 MB)
- `Data/00_shared/raster_data/output_folder/bangladesh/hydrology/`: 21 files (, .tif), 69.5 MB *(written by a notebook)*

### 03_03_multidimentional_data

**03_03_00_introduction_to_multidimentional_data.ipynb**

- `Data/03_spatial_analysis/NetCDF/t2m_2000_2018.nc` (213.2 MB)

**03_03_01_multidimensional_data_mangement.ipynb**

- `Data/00_shared/raster_data/Bangladesh/watertable/watertable_*.tif` (0 B)
- `Data/00_shared/raster_data/Bangladesh/watertable/watertable_*_*.tif` (0 B)
- `Data/00_shared/vector_data/Bangladesh/Shapefiles/BD_Banladesh_BUTM.shp` (2.2 MB)
- `Data/03_spatial_analysis/NetCDF/t2m_2000_2018.nc` (213.2 MB)
- `Data/03_spatial_analysis/NetCDF/output/t2m_*_table_long.nc` (0 B) *(written by a notebook)*
- `Data/03_spatial_analysis/NetCDF/output/t2m_*_table_wide.nc` (0 B) *(written by a notebook)*
- `Data/03_spatial_analysis/NetCDF/output/t2m_all_years_table_long.nc` (741.1 KB) *(written by a notebook)*
- `Data/03_spatial_analysis/NetCDF/output/t2m_all_years_table_records.nc` (26.9 MB) *(written by a notebook)*
- `Data/03_spatial_analysis/NetCDF/output/watertable_1985_2020.nc` (273.8 MB) *(written by a notebook)*
- `Data/03_spatial_analysis/NetCDF/output/Raster_Data/t2m_*_16day_1000m.tif` (0 B) *(written by a notebook)*
- `Data/03_spatial_analysis/NetCDF/output/Tabular_Data/t2m_*_mean_16days.csv` (0 B) *(written by a notebook)*
- `Data/03_spatial_analysis/NetCDF/output/watertable_yearly/watertable_*.nc` (0 B) *(written by a notebook)*

**03_03_02_multidimensional_analysis.ipynb**

- `Data/00_shared/raster_data/Bangladesh/ndvi/ndvi_*_*.tif` (0 B)
- `Data/00_shared/raster_data/Bangladesh/watertable/watertable_*_*.tif` (0 B)
- `Data/03_spatial_analysis/NetCDF/output/multidimensional_analysis/ndvi_*_monthly_mean.nc` (0 B) *(written by a notebook)*
- `Data/03_spatial_analysis/NetCDF/output/multidimensional_analysis/watertable_*_seasonal_mean.nc` (0 B) *(written by a notebook)*
- `Data/03_spatial_analysis/NetCDF/output/multidimensional_analysis/ndvi_2024_monthly_mean/ndvi_*_mean_*.tif` (0 B) *(written by a notebook)*
- `Data/03_spatial_analysis/NetCDF/output/multidimensional_analysis/ndvi_anomaly/ndvi_*_anom_diff_*.tif` (0 B) *(written by a notebook)*
- `Data/03_spatial_analysis/NetCDF/output/multidimensional_analysis/ndvi_anomaly/ndvi_*_anomaly_diff_by_month.nc` (0 B) *(written by a notebook)*
- `Data/03_spatial_analysis/NetCDF/output/multidimensional_analysis/ndvi_anomaly/ndvi_2015_2024_anomaly_diff_by_month.nc` (59.3 MB) *(written by a notebook)*
- `Data/03_spatial_analysis/NetCDF/output/multidimensional_analysis/ndvi_anomaly/ndvi_2015_2024_anomaly_zscore_by_month.nc` (59.6 MB) *(written by a notebook)*
- `Data/03_spatial_analysis/NetCDF/output/multidimensional_analysis/watertable_2020_seasonal_mean/watertable_*_mean_*.tif` (0 B) *(written by a notebook)*

**03_03_03_multidimentioanl_trend_analysis.ipynb**

- `Data/00_shared/raster_data/Bangladesh/watertable/watertable_*.tif` (0 B)
- `Data/03_spatial_analysis/NetCDF/output/multidimensional_trend/watertable_overall_predict_2025_2030.nc` (4.9 MB) *(written by a notebook)*
- `Data/03_spatial_analysis/NetCDF/output/multidimensional_trend/watertable_overall_trend.nc` (9.8 MB) *(written by a notebook)*
- `Data/03_spatial_analysis/NetCDF/output/multidimensional_trend/watertable_seasonal_trend_slope.nc` (7.3 MB) *(written by a notebook)*
- `Data/03_spatial_analysis/NetCDF/output/multidimensional_trend/overall/watertable_overall_MK_pvalue.tif` (729.3 KB) *(written by a notebook)*
- `Data/03_spatial_analysis/NetCDF/output/multidimensional_trend/overall/watertable_overall_MK_slope.tif` (682.1 KB) *(written by a notebook)*

**03_03_04_multidimensional_PCA.ipynb**

- `Data/00_shared/vector_data/Bangladesh/Shapefiles/BD_Banladesh_BUTM.shp` (2.2 MB)
- `Data/03_spatial_analysis/NetCDF/data_2018_fc.nc` (167.9 MB)
- `Data/03_spatial_analysis/NetCDF/output/multidimensional_pca/era_bangladesh_*_monthly_*m.nc` (0 B) *(written by a notebook)*
- `Data/03_spatial_analysis/NetCDF/output/multidimensional_pca/pca/multivar_*_annual_pca.nc` (0 B) *(written by a notebook)*
- `Data/03_spatial_analysis/NetCDF/output/multidimensional_pca/pca/multivar_*_variable_loadings.csv` (0 B) *(written by a notebook)*
- `Data/03_spatial_analysis/NetCDF/output/multidimensional_pca/pca/t2m_*_temporal_pca.nc` (0 B) *(written by a notebook)*

### 03_04_multi_criteria_suitability_analysis

**03_04_01_00_introduction_vector_based_suitability_analysis.ipynb**

- `Data/03_spatial_analysis/MCSM_data/ny_counties.gpkg` (172.0 KB)
- `Data/03_spatial_analysis/MCSM_data/us_counties.geojson` (3.1 MB)
- `Data/03_spatial_analysis/MCSM_data/us_county_summaries.csv` (6.7 MB)
- download: <https://raw.githubusercontent.com/JieYingWu/COVID-19_US_County-level_Summaries/master/data/counties.csv>
- download: <https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json>

**03_04_01_01_deriving_criteria.ipynb**

- `Data/03_spatial_analysis/MCSM_data/ny_counties.gpkg` (172.0 KB)
- `Data/03_spatial_analysis/MCSM_data/ny_criteria.gpkg` (196.0 KB)

**03_04_01_02_categorical_suitability.ipynb**

- `Data/03_spatial_analysis/MCSM_data/ny_categorical.gpkg` (144.0 KB)
- `Data/03_spatial_analysis/MCSM_data/ny_criteria.gpkg` (196.0 KB)

**03_04_01_03_weighted_suitability.ipynb**

- `Data/03_spatial_analysis/MCSM_data/ny_categorical.gpkg` (144.0 KB)
- `Data/03_spatial_analysis/MCSM_data/ny_criteria.gpkg` (196.0 KB)
- `Data/03_spatial_analysis/MCSM_data/ny_suitability_ranking.csv` (12.2 KB)
- `Data/03_spatial_analysis/MCSM_data/ny_weighted_suitability.gpkg` (144.0 KB)

**03_04_02_01_fuzzy_vector_suitability.ipynb**

- `Data/03_spatial_analysis/MCSM_data/ny_criteria.gpkg` (196.0 KB)
- `Data/03_spatial_analysis/MCSM_data/ny_fuzzy_suitability.gpkg` (144.0 KB)
- `Data/03_spatial_analysis/MCSM_data/ny_weighted_suitability.gpkg` (144.0 KB)

**03_04_02_03_fuzzy_raster_suitability.ipynb**

- `Data/03_spatial_analysis/MCSM_data/dem_texas_geo.tif` (269.0 KB)
- `Data/03_spatial_analysis/MCSM_data/dem_texas_utm.tif` (459.3 KB)
- `Data/03_spatial_analysis/MCSM_data/solar_candidate_sites.gpkg` (424.0 KB)
- `Data/03_spatial_analysis/MCSM_data/solar_layers.npz` (4.5 MB)
- `Data/03_spatial_analysis/MCSM_data/solar_suitability_gamma08.tif` (445.9 KB)
- download: <https://raw.githubusercontent.com/mdbartos/pysheds/master/data/dem.tif>

**03_04_02_04_advanced_fuzzy_suitability.ipynb**

- `Data/03_spatial_analysis/MCSM_data/solar_layers.npz` (4.5 MB)

**03_04_03_sensitivity_uncertainty.ipynb**

- `Data/03_spatial_analysis/MCSM_data/solar_candidate_sites.gpkg` (424.0 KB)
- `Data/03_spatial_analysis/MCSM_data/solar_layers.npz` (4.5 MB)
- `Data/03_spatial_analysis/MCSM_data/uncertainty/*.tif` (0 B)
- `Data/03_spatial_analysis/MCSM_data/uncertainty/criteria_sensitivity_ranking.csv` (622 B)
- `Data/03_spatial_analysis/MCSM_data/uncertainty/site_rank_stability.csv` (347 B)
- `Data/03_spatial_analysis/MCSM_data/uncertainty/uncertainty_dashboard.png` (3.0 MB)

### 03_05_network_analysis

**03_05_01_geographic_embedding.ipynb**

- `Data/00_shared/data_spatial_net/roads.graphml` (502.6 KB)
- `Data/00_shared/data_spatial_net/towns.geojson` (4.0 KB)

**03_05_02_osmnx_routing.ipynb**

- `Data/00_shared/data_spatial_net/roads.graphml` (502.6 KB)
- `Data/00_shared/data_spatial_net/study_area.geojson` (402 B)
- `Data/00_shared/data_spatial_net/town_travel_time_min.csv` (7.4 KB)
- `Data/00_shared/data_spatial_net/towns.geojson` (4.0 KB)

**03_05_03_geopandas_networks.ipynb**

- `Data/00_shared/data_spatial_net/roads.graphml` (502.6 KB)
- `Data/00_shared/data_spatial_net/towns.geojson` (4.0 KB)
- `Data/00_shared/data_spatial_net/tracts.geojson` (52.1 KB)

**03_05_04_service_area_access.ipynb**

- `Data/00_shared/data_spatial_net/hospitals.geojson` (1.2 KB)
- `Data/00_shared/data_spatial_net/roads.graphml` (502.6 KB)
- `Data/00_shared/data_spatial_net/towns.geojson` (4.0 KB)
- `Data/00_shared/data_spatial_net/tracts.geojson` (52.1 KB)

**03_05_05_flow_networks.ipynb**

- `Data/00_shared/data_spatial_net/commuting_od.csv` (8.8 KB)
- `Data/00_shared/data_spatial_net/roads.graphml` (502.6 KB)
- `Data/00_shared/data_spatial_net/towns.geojson` (4.0 KB)

### 03_06_segregation_analysis

**03_06_00_introduction_segregation_analysis.ipynb**

- `Data/03_spatial_analysis/data_segregation/builtup_intensity.tif` (306.8 KB)
- `Data/03_spatial_analysis/data_segregation/tracts_*.geojson` (0 B)
- `Data/03_spatial_analysis/data_segregation/tracts_1990_coarse.geojson` (7.3 KB)

**03_06_01_single_group_indices.ipynb**

- `Data/03_spatial_analysis/data_segregation/buffalo/counties.geojson` (64.2 KB)
- `Data/03_spatial_analysis/data_segregation/buffalo/tracts_2000_on2020.geojson` (628.2 KB)
- `Data/03_spatial_analysis/data_segregation/buffalo/tracts_2010_on2020.geojson` (628.2 KB)
- `Data/03_spatial_analysis/data_segregation/buffalo/tracts_2020.geojson` (623.4 KB)

**03_06_02_multigroup_indices.ipynb**

- `Data/03_spatial_analysis/data_segregation/buffalo/counties.geojson` (64.2 KB)
- `Data/03_spatial_analysis/data_segregation/buffalo/tracts_2000_on2020.geojson` (628.2 KB)
- `Data/03_spatial_analysis/data_segregation/buffalo/tracts_2010_on2020.geojson` (628.2 KB)
- `Data/03_spatial_analysis/data_segregation/buffalo/tracts_2020.geojson` (623.4 KB)

**03_06_03_local_segregation.ipynb**

- `Data/03_spatial_analysis/data_segregation/buffalo/counties.geojson` (64.2 KB)
- `Data/03_spatial_analysis/data_segregation/buffalo/tracts_2000_on2020.geojson` (628.2 KB)
- `Data/03_spatial_analysis/data_segregation/buffalo/tracts_2010_on2020.geojson` (628.2 KB)
- `Data/03_spatial_analysis/data_segregation/buffalo/tracts_2020.geojson` (623.4 KB)

**03_06_04_multiscale_profiles.ipynb**

- `Data/03_spatial_analysis/data_segregation/buffalo/blocks_2020_points.geojson` (2.8 MB)
- `Data/03_spatial_analysis/data_segregation/buffalo/counties.geojson` (64.2 KB)
- `Data/03_spatial_analysis/data_segregation/buffalo/tracts_2000_on2020.geojson` (628.2 KB)
- `Data/03_spatial_analysis/data_segregation/buffalo/tracts_2010_on2020.geojson` (628.2 KB)
- `Data/03_spatial_analysis/data_segregation/buffalo/tracts_2020.geojson` (623.4 KB)

**03_06_05_null_models_simulation.ipynb**

- `Data/03_spatial_analysis/data_segregation/buffalo/counties.geojson` (64.2 KB)
- `Data/03_spatial_analysis/data_segregation/buffalo/tracts_2000_on2020.geojson` (628.2 KB)
- `Data/03_spatial_analysis/data_segregation/buffalo/tracts_2010_on2020.geojson` (628.2 KB)
- `Data/03_spatial_analysis/data_segregation/buffalo/tracts_2020.geojson` (623.4 KB)

**03_06_06_inference.ipynb**

- `Data/03_spatial_analysis/data_segregation/buffalo/counties.geojson` (64.2 KB)
- `Data/03_spatial_analysis/data_segregation/buffalo/tracts_2000_on2020.geojson` (628.2 KB)
- `Data/03_spatial_analysis/data_segregation/buffalo/tracts_2010_on2020.geojson` (628.2 KB)
- `Data/03_spatial_analysis/data_segregation/buffalo/tracts_2020.geojson` (623.4 KB)

**03_06_07_decomposition_dynamics.ipynb**

- `Data/03_spatial_analysis/data_segregation/buffalo/`: 7 files (.geojson, .tif), 5.6 MB

### 03_08_capston

**03_08_01_flood_susceptibility_mapping.ipynb**

- `Data/00_shared/vector_data/Bangladesh/Shapefiles/bd_upzila_GCS.shp` (43.5 MB)
- `Data/03_spatial_analysis/Great_plain/Raster/slope.tif` (1012.6 MB) *(by file name)*
- `Data/03_spatial_analysis/Sylhet_flood_susceptibility_mapping/Bangladesh/*.tif` (0 B)
- `Data/03_spatial_analysis/Sylhet_flood_susceptibility_mapping/Bangladesh/DEM.tif` (4.4 MB)
- `Data/03_spatial_analysis/Sylhet_flood_susceptibility_mapping/Bangladesh/bgd-rainfall-subnat-full.csv` (14.4 MB)
- `Data/03_spatial_analysis/Sylhet_flood_susceptibility_mapping/Bangladesh/landuse_2020.tif` (1.1 MB)
- `Data/03_spatial_analysis/Sylhet_flood_susceptibility_mapping/Bangladesh/population_2020.tif` (646.1 KB)
- `Data/03_spatial_analysis/Sylhet_flood_susceptibility_mapping/Bangladesh/IDFILES/DRC_ID.csv` (985 B)
- `Data/03_spatial_analysis/Sylhet_flood_susceptibility_mapping/Bangladesh/IDFILES/LT_ID.csv` (126 B)
- `Data/03_spatial_analysis/Sylhet_flood_susceptibility_mapping/Bangladesh/NDVI/ndvi_2020_*.tif` (0 B)
- `Data/03_spatial_analysis/Sylhet_flood_susceptibility_mapping/Bangladesh/Watertable_2020/watertable_2020_*.tif` (0 B)
- `Data/03_spatial_analysis/Sylhet_flood_susceptibility_mapping/Bangladesh/bgd_hyd_rivers_lged/bgd_hyd_rivers_lged.shp` (85.5 MB)
- `Data/03_spatial_analysis/Sylhet_flood_susceptibility_mapping/Sylhet_sentinel1/sylhet_aoi.geojson` (487 B)
- `Data/03_spatial_analysis/Sylhet_flood_susceptibility_mapping/Sylhet_sentinel1/output/sylhet_after_vh_db.tif` (14.7 MB) *(written by a notebook)*
- `Data/03_spatial_analysis/Sylhet_flood_susceptibility_mapping/Sylhet_sentinel1/output/sylhet_before_vh_db.tif` (14.9 MB) *(written by a notebook)*
- `Data/03_spatial_analysis/Sylhet_flood_susceptibility_mapping/Sylhet_sentinel1/output/sylhet_flood_extent.tif` (303.0 KB) *(written by a notebook)*
- `Data/03_spatial_analysis/capstone_spatial_analysis/great_plains_landslides/ahp_weights.csv` (222 B) *(by file name)*
- `Data/03_spatial_analysis/capstone_spatial_analysis/onondaga/watershed/streams.tif` (1.1 MB) *(by file name)*
- `Data/03_spatial_analysis/capstone_spatial_analysis/onondaga/watershed/twi.tif` (1.1 MB) *(by file name)*
- `Data/03_spatial_analysis/capstone_spatial_analysis/sylhet_flood/`: 8 files (.csv, .gpkg, .gz, .tif), 3.3 MB
- `Data/03_spatial_analysis/capstone_spatial_analysis/sylhet_flood/factors/`: 7 files (.tif), 904.8 KB

**03_08_02_watershed_characterization.ipynb**

- `Data/00_shared/vector_data/USA/NY_County_PROJ.shp` (151.2 KB)
- `Data/03_spatial_analysis/capstone_spatial_analysis/onondaga/watershed/`: 19 files (.csv, .gpkg, .tif), 16.8 MB

**03_08_03_agricultural_suitability.ipynb**

- `Data/00_shared/vector_data/Bangladesh/Shapefiles/bd_upzila_GCS.shp` (43.5 MB)
- `Data/03_spatial_analysis/Wheat_suitability/*.tif` (0 B)
- `Data/03_spatial_analysis/Wheat_suitability/DEM.tif` (4.4 MB)
- `Data/03_spatial_analysis/Wheat_suitability/IDFILES/AEZ_ID.csv` (962 B)
- `Data/03_spatial_analysis/Wheat_suitability/IDFILES/DRC_ID.csv` (985 B)
- `Data/03_spatial_analysis/Wheat_suitability/IDFILES/DROUT_ID.csv` (124 B)
- `Data/03_spatial_analysis/Wheat_suitability/IDFILES/LT_ID.csv` (126 B)
- `Data/03_spatial_analysis/Wheat_suitability/IDFILES/STYPE_ID.csv` (709 B)
- `Data/03_spatial_analysis/Wheat_suitability/Temperature/t2m/t2m_*_*.tif` (0 B)
- `Data/03_spatial_analysis/Wheat_suitability/Temperature/t2m/t2m_2010_049.tif` (1.2 MB)
- `Data/03_spatial_analysis/Wheat_suitability/Temperature/t2m/t2m_20??_001.tif` (0 B)
- `Data/03_spatial_analysis/Wheat_suitability/Watertable_2020/watertable_2020_*.tif` (0 B)
- `Data/03_spatial_analysis/capstone_spatial_analysis/great_plains_landslides/ahp_weights.csv` (222 B) *(by file name)*
- `Data/03_spatial_analysis/capstone_spatial_analysis/onondaga/watershed/slope_deg.tif` (1.1 MB) *(by file name)*
- `Data/03_spatial_analysis/capstone_spatial_analysis/wheat_suitability/`: 10 files (.csv, .gpkg, .tif), 114.7 MB

**03_08_04_landslide_susceptibility.ipynb**

- `Data/03_spatial_analysis/Great_plain/Frg_ID.csv` (677 B)
- `Data/03_spatial_analysis/Great_plain/GP_STATE.shp` (33.8 KB)
- `Data/03_spatial_analysis/Great_plain/gp_grid_new.csv` (105.5 MB)
- `Data/03_spatial_analysis/Great_plain/slope_pos_id.csv` (83 B)
- `Data/03_spatial_analysis/Great_plain/landslides/us_landslides_v3_great_plains.csv` (37.9 MB)
- `Data/03_spatial_analysis/capstone_spatial_analysis/great_plains_landslides/`: 15 files (.csv, .gpkg, .tif), 80.9 MB

**03_08_05_habitat_suitability.ipynb**

- `Data/03_spatial_analysis/Data_and_Code_Forest_Musk_Deer/spe_example.csv` (2.5 KB)
- `Data/03_spatial_analysis/Data_and_Code_Forest_Musk_Deer/current/*.asc` (0 B)
- `Data/03_spatial_analysis/Data_and_Code_Forest_Musk_Deer/current/Dis_road.asc` (82.6 MB)
- `Data/03_spatial_analysis/capstone_spatial_analysis/musk_deer/`: 10 files (.csv, .gpkg, .tif), 22.8 MB

**03_08_06_farmer_networks_case_study.ipynb**

- `Data/00_shared/data_spatial_net/farmer_ties.csv` (24.8 KB)
- `Data/00_shared/data_spatial_net/farmers.geojson` (100.9 KB)

## 04_spatial_statistics · Spatial Statistics

**Dataset folders used in this chapter**

| Dataset folder | Notebooks | Files | Size |
|---|---:|---:|---:|
| `Data/00_shared/data_spatial_stat` | 74 | 18 | 730.8 MB |

### 04_01_point_pattern

**04_01_01_point_pattern_fundamentals.ipynb**

- `Data/00_shared/data_spatial_stat/buffalo_city.shp` (41.1 KB)
- `Data/00_shared/data_spatial_stat/crime_data_2018.csv` (1.7 MB)

**04_01_02_spatial_intensity.ipynb**

- `Data/00_shared/data_spatial_stat/buffalo_city.shp` (41.1 KB)
- `Data/00_shared/data_spatial_stat/crime_data_2018.csv` (1.7 MB)

**04_01_03_second_order_functions.ipynb**

- `Data/00_shared/data_spatial_stat/buffalo_city.shp` (41.1 KB)
- `Data/00_shared/data_spatial_stat/crime_data_2018.csv` (1.7 MB)

**04_01_04_hypothesis_testing.ipynb**

- `Data/00_shared/data_spatial_stat/buffalo_city.shp` (41.1 KB)
- `Data/00_shared/data_spatial_stat/crime_data_2018.csv` (1.7 MB)

**04_01_05_areal_integration_pitfalls.ipynb**

- `Data/00_shared/data_spatial_stat/buffalo_city.shp` (41.1 KB)
- `Data/00_shared/data_spatial_stat/crime_data_2018.csv` (1.7 MB)

### 04_02_spatial_autocorrelation

**04_02_01_concepts_and_foundations.ipynb**

- `Data/00_shared/data_spatial_stat/diabetes_atlantic.shp` (686.5 KB)
- `Data/00_shared/data_spatial_stat/diabetes_northeast.shp` (272.4 KB)

**04_02_02_spatial_weights.ipynb**

- `Data/00_shared/data_spatial_stat/diabetes_northeast.shp` (272.4 KB)

**04_02_03_visualization_exploratory.ipynb**

- `Data/00_shared/data_spatial_stat/diabetes_northeast.shp` (272.4 KB)

**04_02_04_global_autocorrelation.ipynb**

- `Data/00_shared/data_spatial_stat/US_COUNTY.shp` (3.1 MB)
- `Data/00_shared/data_spatial_stat/data_1998_2010_long_lbc.csv` (2.9 MB)
- `Data/00_shared/data_spatial_stat/diabetes_northeast.shp` (272.4 KB)

**04_02_05_local_autocorrelation_lisa.ipynb**

- `Data/00_shared/data_spatial_stat/chr2024_poverty_education.csv` (17.0 KB)
- `Data/00_shared/data_spatial_stat/diabetes_northeast.shp` (272.4 KB)

**04_02_06_sensitivity_pitfalls_exercises.ipynb**

- `Data/00_shared/data_spatial_stat/diabetes_northeast.shp` (272.4 KB)

### 04_03_spatial_regression

**04_03_01_foundations.ipynb**

- `Data/00_shared/data_spatial_stat/diabetes_atlantic.shp` (686.5 KB)

**04_03_02_ols_baseline_diagnostics.ipynb**

- `Data/00_shared/data_spatial_stat/diabetes_atlantic.shp` (686.5 KB)

**04_03_03_model_selection.ipynb**

- `Data/00_shared/data_spatial_stat/diabetes_atlantic.shp` (686.5 KB)

**04_03_04_model_families.ipynb**

- `Data/00_shared/data_spatial_stat/diabetes_atlantic.shp` (686.5 KB)

**04_03_05_comparison_reporting.ipynb**

- `Data/00_shared/data_spatial_stat/diabetes_atlantic.shp` (686.5 KB)

**04_03_06_limits_connections.ipynb**

- `Data/00_shared/data_spatial_stat/diabetes_atlantic.shp` (686.5 KB)

**04_03_07_exercises.ipynb**

- `Data/00_shared/data_spatial_stat/diabetes_atlantic.shp` (686.5 KB)

### 04_04_spatial_cluster

**04_04_01_concepts_overview.ipynb**

- `Data/00_shared/data_spatial_stat/diabetes_northeast.shp` (272.4 KB)

**04_04_02_hotspot_detection.ipynb**

- `Data/00_shared/data_spatial_stat/diabetes_northeast.shp` (272.4 KB)

**04_04_03_density_clustering.ipynb**

- `Data/00_shared/data_spatial_stat/buffalo_city.shp` (41.1 KB)
- `Data/00_shared/data_spatial_stat/crime_data_2018.csv` (1.7 MB)

**04_04_04_scan_statistic.ipynb**

- `Data/00_shared/data_spatial_stat/cancer_areas.shp` (9.8 MB)
- `Data/00_shared/data_spatial_stat/cancer_points.csv` (295.6 KB)
- `Data/00_shared/data_spatial_stat/diabetes_northeast.shp` (272.4 KB)

**04_04_05_regionalisation.ipynb**

- `Data/00_shared/data_spatial_stat/diabetes_northeast.shp` (272.4 KB)

**04_04_06_comparing_validating.ipynb**

- `Data/00_shared/data_spatial_stat/diabetes_northeast.shp` (272.4 KB)

**04_04_07_bridges.ipynb**

- `Data/00_shared/data_spatial_stat/US_COUNTY.shp` (3.1 MB)
- `Data/00_shared/data_spatial_stat/data_1998_2010_long_lbc.csv` (2.9 MB)
- `Data/00_shared/data_spatial_stat/diabetes_northeast.shp` (272.4 KB)

**04_04_08_exercises.ipynb**

- `Data/00_shared/data_spatial_stat/diabetes_northeast.shp` (272.4 KB)

### 04_05_disease_mapping

**04_05_01_introduction_objectives.ipynb**

- `Data/00_shared/data_spatial_stat/diabetes_northeast.shp` (272.4 KB)

**04_05_02_small_area_problem.ipynb**

- `Data/00_shared/data_spatial_stat/diabetes_northeast.shp` (272.4 KB)

**04_05_03_data_preparation.ipynb**

- `Data/00_shared/data_spatial_stat/cancer_areas.shp` (9.8 MB)
- `Data/00_shared/data_spatial_stat/cancer_points.csv` (295.6 KB)
- `Data/00_shared/data_spatial_stat/diabetes_northeast.shp` (272.4 KB)

**04_05_04_raw_estimates.ipynb**

- `Data/00_shared/data_spatial_stat/diabetes_northeast.shp` (272.4 KB)

**04_05_05_empirical_bayes.ipynb**

- `Data/00_shared/data_spatial_stat/diabetes_northeast.shp` (272.4 KB)

**04_05_06_bym_models.ipynb**

- `Data/00_shared/data_spatial_stat/diabetes_northeast.shp` (272.4 KB)

**04_05_07_mcmc_implementation.ipynb**

- `Data/00_shared/data_spatial_stat/diabetes_northeast.shp` (272.4 KB)

**04_05_08_comparing_evaluating.ipynb**

- `Data/00_shared/data_spatial_stat/diabetes_northeast.shp` (272.4 KB)

**04_05_09_shrinkage_assessment.ipynb**

- `Data/00_shared/data_spatial_stat/diabetes_northeast.shp` (272.4 KB)

**04_05_10_reporting_ethics.ipynb**

- `Data/00_shared/data_spatial_stat/diabetes_northeast.shp` (272.4 KB)

**04_05_11_handoffs_exercises.ipynb**

- `Data/00_shared/data_spatial_stat/diabetes_northeast.shp` (272.4 KB)

### 04_06_geographically_weighted_models

**04_06_01_introduction_motivation.ipynb**

- `Data/00_shared/data_spatial_stat/diabetes_atlantic.shp` (686.5 KB)

**04_06_02_data.ipynb**

- `Data/00_shared/data_spatial_stat/data_1998_2010_long_lbc.csv` (2.9 MB)
- `Data/00_shared/data_spatial_stat/diabetes_atlantic.shp` (686.5 KB)
- `Data/00_shared/data_spatial_stat/diabetes_northeast.shp` (272.4 KB)

**04_06_03_kernels_bandwidth.ipynb**

- `Data/00_shared/data_spatial_stat/diabetes_atlantic.shp` (686.5 KB)
- `Data/00_shared/data_spatial_stat/diabetes_northeast.shp` (272.4 KB)

**04_06_04_gwss_exploratory.ipynb**

- `Data/00_shared/data_spatial_stat/diabetes_atlantic.shp` (686.5 KB)

**04_06_05_gwr_inference.ipynb**

- `Data/00_shared/data_spatial_stat/diabetes_atlantic.shp` (686.5 KB)

**04_06_06_advanced_variants.ipynb**

- `Data/00_shared/data_spatial_stat/diabetes_atlantic.shp` (686.5 KB)
- `Data/00_shared/data_spatial_stat/diabetes_northeast.shp` (272.4 KB)

**04_06_07_diagnostics.ipynb**

- `Data/00_shared/data_spatial_stat/diabetes_atlantic.shp` (686.5 KB)

**04_06_08_comparing_models.ipynb**

- `Data/00_shared/data_spatial_stat/diabetes_northeast.shp` (272.4 KB)

**04_06_09_ml_ensemble_models.ipynb**

- `Data/00_shared/data_spatial_stat/diabetes_northeast.shp` (272.4 KB)

**04_06_10_spatiotemporal_multivariate.ipynb**

- `Data/00_shared/data_spatial_stat/data_1998_2010_long_lbc.csv` (2.9 MB)
- `Data/00_shared/data_spatial_stat/diabetes_atlantic.shp` (686.5 KB)
- `Data/00_shared/data_spatial_stat/diabetes_northeast.shp` (272.4 KB)

**04_06_11_deep_network_models.ipynb**

- `Data/00_shared/data_spatial_stat/data_1998_2010_long_lbc.csv` (2.9 MB)
- `Data/00_shared/data_spatial_stat/diabetes_atlantic.shp` (686.5 KB)
- `Data/00_shared/data_spatial_stat/diabetes_northeast.shp` (272.4 KB)

**04_06_12_performance_scaling.ipynb**

- `Data/00_shared/data_spatial_stat/data_1998_2010_long_lbc.csv` (2.9 MB)
- `Data/00_shared/data_spatial_stat/diabetes_atlantic.shp` (686.5 KB)
- `Data/00_shared/data_spatial_stat/diabetes_northeast.shp` (272.4 KB)

**04_06_13_model_selection_troubleshooting.ipynb**

- `Data/00_shared/data_spatial_stat/diabetes_northeast.shp` (272.4 KB)

**04_06_14_exercises.ipynb**

- `Data/00_shared/data_spatial_stat/data_1998_2010_long_lbc.csv` (2.9 MB)
- `Data/00_shared/data_spatial_stat/diabetes_atlantic.shp` (686.5 KB)
- `Data/00_shared/data_spatial_stat/diabetes_northeast.shp` (272.4 KB)

### 04_07_spatial_interpolation

**04_07_01_introduction_concepts.ipynb**

- `Data/00_shared/data_spatial_stat/CA_pm25_2025.csv` (4.1 MB)

**04_07_02_data_preparation.ipynb**

- `Data/00_shared/data_spatial_stat/CA_pm25_2025.csv` (4.1 MB)
- `Data/00_shared/data_spatial_stat/CA_pm25_grid_predictions.csv` (15.3 MB)

**04_07_03_deterministic_interpolation.ipynb**

- `Data/00_shared/data_spatial_stat/meuse.csv` (10.3 KB)
- `Data/00_shared/data_spatial_stat/meuse_grid.csv` (105.5 KB)

**04_07_04_model_validation.ipynb**

- `Data/00_shared/data_spatial_stat/CA_pm25_2025.csv` (4.1 MB)
- `Data/00_shared/data_spatial_stat/CA_pm25_grid_predictions.csv` (15.3 MB)

**04_07_05_covariate_assisted_interpolation.ipynb**

- `Data/00_shared/data_spatial_stat/diabetes_northeast.shp` (272.4 KB)

**04_07_06_areal_interpolation.ipynb**

- `Data/00_shared/data_spatial_stat/cancer_points.csv` (295.6 KB)
- `Data/00_shared/data_spatial_stat/diabetes_northeast.shp` (272.4 KB)

**04_07_07_exercises.ipynb**

- `Data/00_shared/data_spatial_stat/CA_pm25_2025.csv` (4.1 MB)
- `Data/00_shared/data_spatial_stat/diabetes_northeast.shp` (272.4 KB)

### 04_08_spatial_sampling_design

**04_08_01_foundations.ipynb**

- `Data/00_shared/data_spatial_stat/diabetes_usa.csv` (330.5 KB)

**04_08_02_sampling_designs.ipynb**

- `Data/00_shared/data_spatial_stat/diabetes_usa.csv` (330.5 KB)

**04_08_03_estimation_variance.ipynb**

- `Data/00_shared/data_spatial_stat/diabetes_usa.csv` (330.5 KB)

**04_08_04_monitoring_network_design.ipynb**

- `Data/00_shared/data_spatial_stat/CA_pm25_2025.csv` (4.1 MB)

**04_08_05_exercises.ipynb**

- `Data/00_shared/data_spatial_stat/diabetes_usa.csv` (330.5 KB)

### 04_09_spatio_temporal_analysis

**04_09_01_concepts_data.ipynb**

- `Data/00_shared/data_spatial_stat/crime_data_2018.csv` (1.7 MB)
- `Data/00_shared/data_spatial_stat/data_1998_2010_long_lbc.csv` (2.9 MB)

**04_09_02_spacetime_weights_global.ipynb**

- `Data/00_shared/data_spatial_stat/crime_data_2018.csv` (1.7 MB)
- `Data/00_shared/data_spatial_stat/data_1998_2010_long_lbc.csv` (2.9 MB)

**04_09_03_local_emerging_hotspots.ipynb**

- `Data/00_shared/data_spatial_stat/data_1998_2010_long_lbc.csv` (2.9 MB)

**04_09_04_spacetime_point_patterns_density.ipynb**

- `Data/00_shared/data_spatial_stat/crime_data_2018.csv` (1.7 MB)

**04_09_05_spacetime_interpolation.ipynb**

- `Data/00_shared/data_spatial_stat/data_1998_2010_long_lbc.csv` (2.9 MB)

**04_09_06_regression_spacetime_data.ipynb**

- `Data/00_shared/data_spatial_stat/data_1998_2010_long_lbc.csv` (2.9 MB)
- `Data/00_shared/data_spatial_stat/diabetes_northeast.shp` (272.4 KB)

### 04_10_capstone

**04_10_01_health_disparities.ipynb**

- `Data/00_shared/data_spatial_stat/US_COUNTY.shp` (3.1 MB)
- `Data/00_shared/data_spatial_stat/usa_health_atlas_database/usa_health_atlas.sqlite` (154.3 MB)

**04_10_02_voting_pattern.ipynb**

- `Data/00_shared/data_spatial_stat/US_COUNTY.shp` (3.1 MB)
- `Data/00_shared/data_spatial_stat/usa_health_atlas_database/usa_health_atlas.sqlite` (154.3 MB)

**04_10_03_drug_overdose.ipynb**

- `Data/00_shared/data_spatial_stat/US_COUNTY.shp` (3.1 MB)
- `Data/00_shared/data_spatial_stat/drug_overdose_all_2014_2018_updated_no_NA.csv` (445.8 KB)

**04_10_04_spatial_temporal_crime_analysis_NYC.ipynb**

- `Data/00_shared/data_spatial_stat/NYC_Crime_data/nyc_crime.sqlite` (26.4 MB)
- `Data/00_shared/data_spatial_stat/NYC_Crime_data/nyc_crime_spatiotemporal.gpkg` (76.0 MB)

**04_10_05_spatial_inequality_economic_deprivation.ipynb**

- `Data/00_shared/data_spatial_stat/US_COUNTY.shp` (3.1 MB)
- `Data/00_shared/data_spatial_stat/usa_health_atlas_database/usa_health_atlas.sqlite` (154.3 MB)

## 05_geostatistics · Geostatistics

**Dataset folders used in this chapter**

| Dataset folder | Notebooks | Files | Size |
|---|---:|---:|---:|
| `Data/00_shared/data_geostatistics` | 47 | 88 | 219.3 MB |

### 05_01_foundation

**05_00_getting_started.ipynb**

- `Data/00_shared/data_geostatistics/meuse.csv` (10.3 KB)
- `Data/00_shared/data_geostatistics/meuse_grid.csv` (105.5 KB)

**05_01_variogram_modeling.ipynb**

- `Data/00_shared/data_geostatistics/gp_data_5000.csv` (715.0 KB)
- `Data/00_shared/data_geostatistics/jura_data.csv` (20.3 KB)

### 05_02_kriging

**05_02_01_simple_kriging.ipynb**

- `Data/00_shared/data_geostatistics/gp_data_5000.csv` (715.0 KB)
- `Data/00_shared/data_geostatistics/jura_data.csv` (20.3 KB)
- `Data/00_shared/data_geostatistics/jura_grid.csv` (220.8 KB)

**05_02_02_ordinary_kriging.ipynb**

- `Data/00_shared/data_geostatistics/gp_data_5000.csv` (715.0 KB)
- `Data/00_shared/data_geostatistics/jura_data.csv` (20.3 KB)
- `Data/00_shared/data_geostatistics/jura_grid.csv` (220.8 KB)

**05_02_03_universal_kriging.ipynb**

- `Data/00_shared/data_geostatistics/gp_data_5000.csv` (715.0 KB)
- `Data/00_shared/data_geostatistics/jura_data.csv` (20.3 KB)
- `Data/00_shared/data_geostatistics/jura_grid.csv` (220.8 KB)

**05_02_04_cokriging.ipynb**

- `Data/00_shared/data_geostatistics/jura_data.csv` (20.3 KB)
- `Data/00_shared/data_geostatistics/jura_grid.csv` (220.8 KB)

**05_02_05_gp_data_5000_jax_comparison.ipynb**

- `Data/00_shared/data_geostatistics/Data_GP/gp_data_5000.csv` (715.0 KB)

### 05_03_regression_kriging

**05_03_01_regression_kriging_scikit_learn.ipynb**

- `Data/00_shared/data_geostatistics/GP_STATE.shp` (33.8 KB)
- `Data/00_shared/data_geostatistics/gp_data_5000.csv` (715.0 KB)
- `Data/00_shared/data_geostatistics/gp_grid.csv` (1.3 MB)
- `Data/00_shared/data_geostatistics/gp_prediction_RK.csv` (4.2 MB)

**05_03_02_regression_kriging_h2o.ipynb**

- `Data/00_shared/data_geostatistics/GP_STATE.shp` (33.8 KB)
- `Data/00_shared/data_geostatistics/gp_data_5000.csv` (715.0 KB)
- `Data/00_shared/data_geostatistics/gp_grid.csv` (1.3 MB)
- `Data/00_shared/data_geostatistics/gp_prediction_RK_h2o.csv` (3.2 MB)

**05_03_03_regression_kriging_pytorch.ipynb**

- `Data/00_shared/data_geostatistics/GP_STATE.shp` (33.8 KB)
- `Data/00_shared/data_geostatistics/gp_data_5000.csv` (715.0 KB)
- `Data/00_shared/data_geostatistics/gp_grid.csv` (1.3 MB)
- `Data/00_shared/data_geostatistics/gp_prediction_RK_pytorch.csv` (2.4 MB)

**05_03_04_regression_kriging_dnn_tf.ipynb**

- `Data/00_shared/data_geostatistics/GP_STATE.shp` (33.8 KB)
- `Data/00_shared/data_geostatistics/gp_data_5000.csv` (715.0 KB)
- `Data/00_shared/data_geostatistics/gp_grid.csv` (1.3 MB)
- `Data/00_shared/data_geostatistics/gp_prediction_RK_tf.csv` (2.4 MB)

**05_03_05_regression_kriging_gnn.ipynb**

- `Data/00_shared/data_geostatistics/GP_STATE.shp` (33.8 KB)
- `Data/00_shared/data_geostatistics/gp_data_5000.csv` (715.0 KB)
- `Data/00_shared/data_geostatistics/gp_grid.csv` (1.3 MB)
- `Data/00_shared/data_geostatistics/gp_prediction_RK_gnn.csv` (2.8 MB)

### 05_04_specialized_kriging

**05_04_01_indicator_kriging.ipynb**

- `Data/00_shared/data_geostatistics/meuse.csv` (10.3 KB)
- `Data/00_shared/data_geostatistics/meuse_grid.csv` (105.5 KB)

**05_04_02_kriging_3d.ipynb**

- `Data/00_shared/data_geostatistics/GSLIB_data/cluster.dat` (3.9 KB)

**05_04_03_soft_kriging.ipynb**

- `Data/00_shared/data_geostatistics/jura_data.csv` (20.3 KB)
- `Data/00_shared/data_geostatistics/jura_grid.csv` (220.8 KB)

**05_04_04_disjunctive_kriging.ipynb**

- `Data/00_shared/data_geostatistics/jura_data.csv` (20.3 KB)
- `Data/00_shared/data_geostatistics/jura_grid.csv` (220.8 KB)

**05_04_05_factorial_kriging.ipynb**

- `Data/00_shared/data_geostatistics/jura_data.csv` (20.3 KB)
- `Data/00_shared/data_geostatistics/jura_grid.csv` (220.8 KB)

### 05_05_spatial_temporal_kriging

**05_05_01_kriging_st.ipynb**

- `Data/00_shared/data_geostatistics/CA_pm25_2025.csv` (4.1 MB)
- `Data/00_shared/data_geostatistics/CA_pm25_grid_predictions.csv` (15.3 MB)
- `Data/00_shared/data_geostatistics/US_STATE.shp` (1.1 MB)

**05_05_02_kriging_cnn_lstm.ipynb**

- `Data/00_shared/data_geostatistics/CA_pm25_2025.csv` (4.1 MB)
- `Data/00_shared/data_geostatistics/CA_pm25_grid_predictions.csv` (15.3 MB)
- `Data/00_shared/data_geostatistics/US_STATE.shp` (1.1 MB)

**05_05_03_kriging_stgnn.ipynb**

- `Data/00_shared/data_geostatistics/CA_pm25_2025.csv` (4.1 MB)
- `Data/00_shared/data_geostatistics/CA_pm25_grid_predictions.csv` (15.3 MB)
- `Data/00_shared/data_geostatistics/US_STATE.shp` (1.1 MB)

**05_05_04_kriging_stgtn.ipynb**

- `Data/00_shared/data_geostatistics/CA_pm25_2025.csv` (4.1 MB)
- `Data/00_shared/data_geostatistics/CA_pm25_grid_predictions.csv` (15.3 MB)
- `Data/00_shared/data_geostatistics/US_STATE.shp` (1.1 MB)

### 05_06_poisson_kriging

**05_06_01_poisson_kriging_ata_atp.ipynb**

- `Data/00_shared/data_geostatistics/US_COUNTY.shp` (3.1 MB)
- `Data/00_shared/data_geostatistics/cancer_areas.shp` (9.8 MB)
- `Data/00_shared/data_geostatistics/cancer_data.gpkg` (10.6 MB)
- `Data/00_shared/data_geostatistics/usa_diabetes_data.csv` (331.8 KB)

**05_06_02_poisson_cokriging.ipynb**

- `Data/00_shared/data_geostatistics/US_COUNTY.shp` (3.1 MB)
- `Data/00_shared/data_geostatistics/diabetes_northeast.csv` (24.2 KB)
- `Data/00_shared/data_geostatistics/diabetes_northeast.shp` (272.4 KB)
- `Data/00_shared/data_geostatistics/usa_diabetes_data.csv` (331.8 KB)

**05_06_03_poisson_kriging_st.ipynb**

- `Data/00_shared/data_geostatistics/COUNTY_ATLANTIC.shp` (686.5 KB)
- `Data/00_shared/data_geostatistics/LBC_1998_2012_USA.csv` (4.8 MB)
- `Data/00_shared/data_geostatistics/US_COUNTY.shp` (3.1 MB)
- `Data/00_shared/data_geostatistics/lbc_atlantic.csv` (120.0 KB)
- `Data/00_shared/data_geostatistics/lbc_rate.csv` (90.5 KB)
- `Data/00_shared/data_geostatistics/usa_diabetes_data.csv` (331.8 KB)

### 05_07_geostatistical_simulation

**05_07_01_sgsim.ipynb**

- `Data/00_shared/data_geostatistics/meuse.csv` (10.3 KB)
- `Data/00_shared/data_geostatistics/meuse_grid.csv` (105.5 KB)

**05_07_02_sgcosim.ipynb**

- `Data/00_shared/data_geostatistics/jura_data.csv` (20.3 KB) *(by file name)*
- `Data/00_shared/data_geostatistics/jura_grid.csv` (220.8 KB) *(by file name)*

**05_07_03_sisim.ipynb**

- `Data/00_shared/data_geostatistics/meuse.csv` (10.3 KB)
- `Data/00_shared/data_geostatistics/meuse_grid.csv` (105.5 KB)

**05_07_04_sicosim.ipynb**

- `Data/00_shared/data_geostatistics/jura_data.csv` (20.3 KB) *(by file name)*
- `Data/00_shared/data_geostatistics/jura_grid.csv` (220.8 KB) *(by file name)*

**05_07_05_dssim.ipynb**

- `Data/00_shared/data_geostatistics/jura_data.csv` (20.3 KB) *(by file name)*
- `Data/00_shared/data_geostatistics/jura_grid.csv` (220.8 KB) *(by file name)*

**05_07_06_sasim.ipynb**

- `Data/00_shared/data_geostatistics/jura_data.csv` (20.3 KB) *(by file name)*
- `Data/00_shared/data_geostatistics/jura_grid.csv` (220.8 KB) *(by file name)*

**05_07_07_pgs.ipynb**

- `Data/00_shared/data_geostatistics/jura_data.csv` (20.3 KB) *(by file name)*
- `Data/00_shared/data_geostatistics/jura_grid.csv` (220.8 KB) *(by file name)*

**05_07_08_mps.ipynb**

- `Data/00_shared/data_geostatistics/jura_data.csv` (20.3 KB) *(by file name)*
- `Data/00_shared/data_geostatistics/jura_grid.csv` (220.8 KB) *(by file name)*

### 05_08_bayesian_kriging

**05_08_01_empirical_bayesian_kriging.ipynb**

- `Data/00_shared/data_geostatistics/jura_data.csv` (20.3 KB)
- `Data/00_shared/data_geostatistics/jura_grid.csv` (220.8 KB)

**05_08_02_bayesian_kriging.ipynb**

- `Data/00_shared/data_geostatistics/jura_data.csv` (20.3 KB) *(by file name)*
- `Data/00_shared/data_geostatistics/jura_grid.csv` (220.8 KB) *(by file name)*

**05_08_03_bayesian_glm_kriging.ipynb**

- `Data/00_shared/data_geostatistics/COUNTY_ATLANTIC.shp` (686.5 KB) *(by file name)*
- `Data/00_shared/data_geostatistics/diabetes_atlantic.csv` (73.6 KB) *(by file name)*
- `Data/00_shared/data_geostatistics/lbc_rate.csv` (90.5 KB) *(by file name)*

**05_08_04_bayesian_transformed_kriging.ipynb**

- `Data/00_shared/data_geostatistics/jura_data.csv` (20.3 KB) *(by file name)*
- `Data/00_shared/data_geostatistics/jura_grid.csv` (220.8 KB) *(by file name)*
- `Data/00_shared/data_geostatistics/meuse.csv` (10.3 KB) *(by file name)*

**05_08_05_conjugate_bayesian_kriging.ipynb**

- `Data/00_shared/data_geostatistics/jura_data.csv` (20.3 KB) *(by file name)*
- `Data/00_shared/data_geostatistics/jura_grid.csv` (220.8 KB) *(by file name)*

**05_08_06_bayesian_kriging_function.ipynb**

- `Data/00_shared/data_geostatistics/jura_data.csv` (20.3 KB) *(by file name)*
- `Data/00_shared/data_geostatistics/jura_grid.csv` (220.8 KB) *(by file name)*

**05_08_07_correlation_function.ipynb**

- `Data/00_shared/data_geostatistics/jura_data.csv` (20.3 KB) *(by file name)*

### 05_09_capstone

**05_09_01_groundwater_arsenic_risk.ipynb**

- `Data/00_shared/data_geostatistics/BGS_BD/BD_Banladesh_BUTM.shp` (2.2 MB)
- `Data/00_shared/data_geostatistics/BGS_BD/bd_grid.csv` (139.7 KB)
- `Data/00_shared/data_geostatistics/BGS_BD/bgs_geochemical.csv` (792.3 KB)

**05_09_02_arsenic_cancer_risk.ipynb**

- `Data/00_shared/data_geostatistics/Arsenic - Tala/tala_grid.csv` (1.5 MB)
- `Data/00_shared/data_geostatistics/Arsenic - Tala/tala_union.shp` (200.4 KB)
- `Data/00_shared/data_geostatistics/Arsenic - Tala/tala_union_centriod.dbf` (1.8 KB)
- `Data/00_shared/data_geostatistics/Arsenic - Tala/water_soil_rice_arsenic_data.csv` (65.7 KB)

**05_09_03_uncertinity_risk_mapping.ipynb**

- `Data/00_shared/data_geostatistics/Arsenic - Tala/All_data.csv` (49.0 KB)
- `Data/00_shared/data_geostatistics/Arsenic - Tala/Tala_grid.csv` (3.2 MB)
- `Data/00_shared/data_geostatistics/Arsenic - Tala/tala.csv` (75.6 KB)
- `Data/00_shared/data_geostatistics/Arsenic - Tala/tala_union.shp` (200.4 KB)

**05_09_04_cd_mapping_in_usa.ipynb**

- `Data/00_shared/data_geostatistics/US_STATE.shp` (1.1 MB)
- `Data/00_shared/data_geostatistics/USA_Geochemical/usa_geochemical.csv` (1.3 MB)
- `Data/00_shared/data_geostatistics/USA_Geochemical/usa_geochemical_meta_data.csv` (1.6 KB)

**05_09_05_poisson_cokriging_diabetes_obesity.ipynb**

- `Data/00_shared/data_geostatistics/US_COUNTY.shp` (3.1 MB)
- `Data/00_shared/data_geostatistics/US_STATE.shp` (1.1 MB)
- `Data/00_shared/data_geostatistics/USA_Data/diabetes_obesity_county_2020_2024.csv` (1.3 MB)

**05_09_06_micronutrient_deficiency_mapping_Bangladesh.ipynb**

- `Data/00_shared/data_geostatistics/BD_Data/BD_Banladesh_BUTM.shp` (2.2 MB)
- `Data/00_shared/data_geostatistics/BD_Data/bd_grid.csv` (18.0 MB)
- `Data/00_shared/data_geostatistics/BD_Data/bd_soil.csv` (9.6 MB)

**05_09_07_poisson_kriging_STD.ipynb**

- `Data/00_shared/data_geostatistics/US_COUNTY.shp` (3.1 MB)
- `Data/00_shared/data_geostatistics/USA_Data/Ca_las-sql-2026-11-14.csv` (89.6 KB)
- `Data/00_shared/data_geostatistics/USA_Data/ca_stds_by_disease_county_year_sex.csv` (671.3 KB)

**05_09_08_spatial_temporal_forcasting_groundwater_BD.ipynb**

- `Data/00_shared/data_geostatistics/BD_Raj/RAJ_Raster/raj_DEM_BUTM.tif` (1.2 MB)
- `Data/00_shared/data_geostatistics/BD_Raj/RAJ_Raster/Cube_1km/*_2010_2018.tif` (0 B)
- `Data/00_shared/data_geostatistics/BD_Raj/RAJ_Raster/Water_table/wt_*_*.tif` (0 B)
- `Data/00_shared/data_geostatistics/BD_Raj/RAJ_Raster/Water_table_forecast/wt_forecast_2018_*_*.tif` (0 B)
- `Data/00_shared/data_geostatistics/BD_Raj/Raj_Groundwater_CSV/`: 37 files (.csv), 15.1 MB
- `Data/00_shared/data_geostatistics/BD_Raj/Shapefiles/raj_division_BUTM.shp` (491.5 KB)

## 06_spatialDB · Spatial Databases

**Dataset folders used in this chapter**

| Dataset folder | Notebooks | Files | Size |
|---|---:|---:|---:|
| `Data/06_spatialDB/atlas` | 14 | 1 | 42.9 MB |
| `Data/00_shared/data_geostatistics` | 3 | 2 | 5.0 MB |
| `Data/06_spatialDB/nyc-taxi-tiny` | 3 | 25 | 5.8 MB |
| `Data/00_shared/vector_data` | 3 | 1 | 526.6 KB |
| `Data/00_shared/data_gdal` | 1 | 1 | 11.8 KB |
| `Data/00_shared/data_remote_sensing` | 1 | 1 | 15.8 KB |
| `Data/00_shared/raster_data` | 1 | 1 | 151.2 KB |
| `Data/00_shared/data_web_cloud_gis` | 1 | 1 | 44.9 KB |
| `Data/00_shared/data_spatial_net` | 1 | 4 | 559.9 KB |
| `Data/06_spatialDB` | 1 | 13 | 606.3 MB |

### 06_01_foundation

**06_01_01_spatial_database_concepts.ipynb**

- `Data/00_shared/data_geostatistics/Data_GP/RASTER/DEM/Onondaga_DEM.tif` (655.7 KB) *(by file name)*
- `Data/06_spatialDB/atlas/assets/geojson_counties_fips.json` (3.1 MB) *(by file name)*
- download: <https://raw.githubusercontent.com/zia207/USA-Health-Data-Atlas/main>

**06_01_02_environment_setup.ipynb**

- `Data/06_spatialDB/atlas/assets/geojson_counties_fips.json` (3.1 MB) *(by file name)*
- download: <https://raw.githubusercontent.com/zia207/USA-Health-Data-Atlas/main>

**06_01_03_filebased_spatial_database.ipynb**

- `Data/06_spatialDB/atlas/assets/geojson_counties_fips.json` (3.1 MB) *(by file name)*
- download: <https://raw.githubusercontent.com/zia207/USA-Health-Data-Atlas/main>

### 06_02_spatial_database_core

**06_02_01_loading_exporting_data.ipynb**

- `Data/00_shared/data_geostatistics/Data_GP/RASTER/DEM/Onondaga_DEM.tif` (655.7 KB) *(by file name)*
- `Data/00_shared/data_geostatistics/USA_Data/US_COUNTY.shp` (3.1 MB) *(by file name)*
- `Data/06_spatialDB/atlas/assets/geojson_counties_fips.json` (3.1 MB) *(by file name)*
- download: <https://raw.githubusercontent.com/zia207/USA-Health-Data-Atlas/main>
- download: <https://www2.census.gov/geo/tiger/TIGER2024/COUNTY/tl_2024_us_county.zip>
- download: <https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson>
- download: <https://raw.githubusercontent.com/gboeing/osmnx/main/tests/input_data>
- download: <https://download.geofabrik.de/north-america/us/new-york-latest.osm.pbf>

**06_02_02_spatial_sql_fundamentals.ipynb**

- `Data/00_shared/vector_data/USA/Ononda_Street_PROJ.shp` (175.5 KB) *(by file name)*
- `Data/06_spatialDB/atlas/assets/geojson_counties_fips.json` (3.1 MB) *(by file name)*
- download: <https://raw.githubusercontent.com/zia207/USA-Health-Data-Atlas/main>

**06_02_03_spatial_relationship_joins.ipynb**

- `Data/06_spatialDB/atlas/assets/geojson_counties_fips.json` (3.1 MB) *(by file name)*
- download: <https://raw.githubusercontent.com/zia207/USA-Health-Data-Atlas/main>

**06_02_04_geometry_processing.ipynb**

- `Data/00_shared/raster_data/USA/NY_County_PROJ.shp` (151.2 KB) *(by file name)*
- `Data/00_shared/vector_data/USA/Ononda_Street_PROJ.shp` (175.5 KB) *(by file name)*
- `Data/06_spatialDB/atlas/assets/geojson_counties_fips.json` (3.1 MB) *(by file name)*
- download: <https://raw.githubusercontent.com/zia207/USA-Health-Data-Atlas/main>

**06_02_05_indexing_and_performance.ipynb**

- `Data/06_spatialDB/atlas/assets/geojson_counties_fips.json` (3.1 MB) *(by file name)*
- `Data/06_spatialDB/nyc-taxi-tiny/year=2009/month=1/part-0.parquet` (781.8 KB) *(by file name)*
- download: <https://raw.githubusercontent.com/zia207/USA-Health-Data-Atlas/main>
- download: <https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson>

### 06_03_advanced_spatial_databases

**06_03_01_raster_PostGIS.ipynb**

- `Data/00_shared/data_geostatistics/Data_GP/RASTER/DEM/Onondaga_DEM.tif` (655.7 KB) *(by file name)*
- `Data/00_shared/data_remote_sensing/Data_Ithaca/HLS_Sentinel-2/HLS.S30.T18TUN.2025246T155701.v2.0/item.json` (15.8 KB) *(by file name)*
- `Data/00_shared/data_web_cloud_gis/s2_syracuse_2025/S2B_18TVN_20250919_0_L2A/scl.tif` (44.9 KB) *(by file name)*
- `Data/06_spatialDB/atlas/assets/geojson_counties_fips.json` (3.1 MB) *(by file name)*
- download: <https://raw.githubusercontent.com/zia207/USA-Health-Data-Atlas/main>

**06_03_02_network_analysis_pgRouting.ipynb**

- `Data/00_shared/data_spatial_net/hospitals.geojson` (1.2 KB) *(by file name)*
- `Data/00_shared/data_spatial_net/roads.graphml` (502.6 KB) *(by file name)*
- `Data/00_shared/data_spatial_net/towns.geojson` (4.0 KB) *(by file name)*
- `Data/00_shared/data_spatial_net/tracts.geojson` (52.1 KB) *(by file name)*
- `Data/00_shared/vector_data/USA/Ononda_Street_PROJ.shp` (175.5 KB) *(by file name)*
- `Data/06_spatialDB/atlas/assets/geojson_counties_fips.json` (3.1 MB) *(by file name)*
- download: <https://raw.githubusercontent.com/zia207/USA-Health-Data-Atlas/main>
- download: <https://raw.githubusercontent.com/gboeing/osmnx/main/tests/input_data>

**06_03_03_spatiotemporal_data.ipynb**

- `Data/00_shared/data_gdal/gtfs/gtfs_extract.zip` (11.8 KB) *(by file name)*
- `Data/06_spatialDB/atlas/assets/geojson_counties_fips.json` (3.1 MB) *(by file name)*
- `Data/06_spatialDB/nyc-taxi-tiny/year=2009/month=1/part-0.parquet` (781.8 KB) *(by file name)*
- download: <https://raw.githubusercontent.com/zia207/USA-Health-Data-Atlas/main>

**06_03_04_in_database_spatial_analytics.ipynb**

- `Data/06_spatialDB/atlas/assets/geojson_counties_fips.json` (3.1 MB) *(by file name)*
- download: <https://raw.githubusercontent.com/zia207/USA-Health-Data-Atlas/main>

**06_03_05_DuckDB_spatial_geoParquet.ipynb**

- `Data/06_spatialDB/`: 13 files (.parquet), 606.3 MB
- `Data/06_spatialDB/atlas/assets/geojson_counties_fips.json` (3.1 MB) *(by file name)*
- `Data/06_spatialDB/nyc-taxi-tiny/year=2009/month=1/part-0.parquet` (781.8 KB) *(by file name)*
- `Data/06_spatialDB/nyc-taxi-tiny/year=2018/month=1/part-0.parquet` (224.8 KB) *(by glob pattern)*
- `Data/06_spatialDB/nyc-taxi-tiny/year=2018/month=10/part-0.parquet` (230.0 KB) *(by glob pattern)*
- `Data/06_spatialDB/nyc-taxi-tiny/year=2018/month=11/part-0.parquet` (211.6 KB) *(by glob pattern)*
- `Data/06_spatialDB/nyc-taxi-tiny/year=2018/month=12/part-0.parquet` (212.1 KB) *(by glob pattern)*
- `Data/06_spatialDB/nyc-taxi-tiny/year=2018/month=2/part-0.parquet` (218.8 KB) *(by glob pattern)*
- `Data/06_spatialDB/nyc-taxi-tiny/year=2018/month=3/part-0.parquet` (242.7 KB) *(by glob pattern)*
- `Data/06_spatialDB/nyc-taxi-tiny/year=2018/month=4/part-0.parquet` (239.7 KB) *(by glob pattern)*
- `Data/06_spatialDB/nyc-taxi-tiny/year=2018/month=5/part-0.parquet` (238.2 KB) *(by glob pattern)*
- `Data/06_spatialDB/nyc-taxi-tiny/year=2018/month=6/part-0.parquet` (226.9 KB) *(by glob pattern)*
- `Data/06_spatialDB/nyc-taxi-tiny/year=2018/month=7/part-0.parquet` (201.8 KB) *(by glob pattern)*
- `Data/06_spatialDB/nyc-taxi-tiny/year=2018/month=8/part-0.parquet` (203.4 KB) *(by glob pattern)*
- `Data/06_spatialDB/nyc-taxi-tiny/year=2018/month=9/part-0.parquet` (209.0 KB) *(by glob pattern)*
- `Data/06_spatialDB/nyc-taxi-tiny/year=2021/month=1/part-0.parquet` (47.3 KB) *(by glob pattern)*
- `Data/06_spatialDB/nyc-taxi-tiny/year=2021/month=10/part-0.parquet` (100.8 KB) *(by glob pattern)*
- `Data/06_spatialDB/nyc-taxi-tiny/year=2021/month=11/part-0.parquet` (101.6 KB) *(by glob pattern)*
- `Data/06_spatialDB/nyc-taxi-tiny/year=2021/month=12/part-0.parquet` (95.7 KB) *(by glob pattern)*
- `Data/06_spatialDB/nyc-taxi-tiny/year=2021/month=2/part-0.parquet` (47.0 KB) *(by glob pattern)*
- `Data/06_spatialDB/nyc-taxi-tiny/year=2021/month=3/part-0.parquet` (61.0 KB) *(by glob pattern)*
- `Data/06_spatialDB/nyc-taxi-tiny/year=2021/month=4/part-0.parquet` (68.9 KB) *(by glob pattern)*
- `Data/06_spatialDB/nyc-taxi-tiny/year=2021/month=5/part-0.parquet` (77.4 KB) *(by glob pattern)*
- `Data/06_spatialDB/nyc-taxi-tiny/year=2021/month=6/part-0.parquet` (85.6 KB) *(by glob pattern)*
- `Data/06_spatialDB/nyc-taxi-tiny/year=2021/month=7/part-0.parquet` (85.6 KB) *(by glob pattern)*
- `Data/06_spatialDB/nyc-taxi-tiny/year=2021/month=8/part-0.parquet` (84.9 KB) *(by glob pattern)*
- `Data/06_spatialDB/nyc-taxi-tiny/year=2021/month=9/part-0.parquet` (89.5 KB) *(by glob pattern)*
- download: <https://raw.githubusercontent.com/zia207/USA-Health-Data-Atlas/main>
- download: <https://stac.overturemaps.org/catalog.json>

**06_03_06_bigda_data_cloud_spatial_engines.ipynb**

- `Data/06_spatialDB/atlas/assets/geojson_counties_fips.json` (3.1 MB) *(by file name)*
- download: <https://raw.githubusercontent.com/zia207/USA-Health-Data-Atlas/main>

## 07_web_cloud_gis · Web and Cloud GIS

**Dataset folders used in this chapter**

| Dataset folder | Notebooks | Files | Size |
|---|---:|---:|---:|
| `Data/00_shared/data_web_cloud_gis` | 11 | 86 | 1.4 GB |
| `Data/00_shared/vector_data` | 7 | 10 | 822.2 MB |
| `Data/00_shared/data_remote_sensing` | 4 | 18 | 22.0 MB |
| `Data/00_shared/tabular_data` | 2 | 1 | 17.6 MB |
| `Data/00_shared/data_spatial_stat` | 1 | 1 | 4.1 MB |

**07_00_introduction_web_cloud_gis.ipynb**

- `Data/00_shared/vector_data/USA/NY_County_GCS.shp` (151.2 KB)
- download: <https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/18/T/UN/2025/8/S2C_18TUN_20250802_0_L2A/B04.tif>

**07_01_cloud_native_raster.ipynb**

- `Data/00_shared/data_remote_sensing/Data_Ithaca/HLS_Sentinel-2/HLS.S30.T18TUN.2025246T155701.v2.0/item.json` (15.8 KB) *(by file name)*
- `Data/00_shared/data_remote_sensing/Data_Ithaca/HLS_Sentinel-2/HLS.S30.T18TUN.2025246T155701.v2.0/red.tif` (10.4 MB) *(by file name)*
- `Data/00_shared/data_remote_sensing/Data_Ithaca/NAIP/naip_2026-08-17_2026-10-16/stack.tif` (11.5 MB) *(by file name)*
- `Data/00_shared/data_web_cloud_gis/08_01_outputs/ndvi_syracuse_*.tif` (0 B)
- `Data/00_shared/data_web_cloud_gis/08_01_outputs/stack_*_*.tif` (0 B)
- `Data/00_shared/data_web_cloud_gis/08_01_outputs/stack_deflate_2.tif` (15.2 MB)
- `Data/00_shared/data_web_cloud_gis/08_01_outputs/stack_web_optimized.tif` (8.0 MB)
- `Data/00_shared/data_web_cloud_gis/08_01_outputs/syracuse_ndvi_items.parquet` (11.3 KB)
- `Data/00_shared/data_web_cloud_gis/08_01_outputs/syracuse_s2_2025.zarr` (72.5 MB)
- `Data/00_shared/data_web_cloud_gis/08_01_outputs/syracuse_s2_2025.zarr/*` (0 B)
- `Data/00_shared/data_web_cloud_gis/s2_syracuse_2025/*` (0 B)
- `Data/00_shared/data_web_cloud_gis/s2_syracuse_2025/request.json` (951 B)
- `Data/00_shared/data_web_cloud_gis/s2_syracuse_2025/S2B_18TVN_20250919_0_L2A/scl.tif` (44.9 KB) *(by file name)*

**07_02_cloud_native_vector.ipynb**

- `Data/00_shared/data_web_cloud_gis/08_02_outputs/`: 8 files (.fgb, .geojson, .gpkg, .parquet, .pmtiles), 992.1 MB
- `Data/00_shared/vector_data/Bangladesh/Shapefiles/bd_division_GCS.shp` (5.2 MB)
- `Data/00_shared/vector_data/Bangladesh/Shapefiles/bd_union_GCS.*` (0 B)
- `Data/00_shared/vector_data/Bangladesh/Shapefiles/bd_union_GCS.shp` (118.8 MB)

**07_03_spatial_sql.ipynb**

- `Data/00_shared/data_web_cloud_gis/08_03_outputs/districts.fgb` (16.3 MB)
- `Data/00_shared/data_web_cloud_gis/08_03_outputs/districts.gpkg` (16.4 MB)
- `Data/00_shared/data_web_cloud_gis/08_03_outputs/districts_duckdb.parquet` (14.0 MB)
- `Data/00_shared/data_web_cloud_gis/08_03_outputs/districts_geolattice.parquet` (14.2 MB)
- `Data/00_shared/vector_data/Bangladesh/bd_arsenic_data.csv` (39.6 KB)
- `Data/00_shared/vector_data/Bangladesh/bd_data.csv` (58.5 KB)
- `Data/00_shared/vector_data/Bangladesh/bd_district_rice_production_2018_2019_BTM.shp` (16.2 MB)
- `Data/00_shared/vector_data/Bangladesh/bd_poverty.csv` (7.8 KB)
- `Data/00_shared/vector_data/Bangladesh/bd_upazila_indicators.csv` (405.1 KB)
- `Data/00_shared/vector_data/Bangladesh/Shapefiles/bd_union_GCS.shp` (118.8 MB)
- `Data/00_shared/vector_data/Bangladesh/Shapefiles/bd_upzila_GCS.shp` (43.5 MB)

**07_04_interactive_web_maps.ipynb**

- `Data/00_shared/data_web_cloud_gis/08_01_outputs/ndvi_syracuse_*.tif` (0 B)
- `Data/00_shared/data_web_cloud_gis/08_01_outputs/ndvi_syracuse_20250723.tif` (8.5 MB)
- `Data/00_shared/data_web_cloud_gis/08_01_outputs/ndvi_syracuse_20250919.tif` (8.8 MB)
- `Data/00_shared/data_web_cloud_gis/08_03_outputs/districts.gpkg` (16.4 MB)
- `Data/00_shared/data_web_cloud_gis/08_04_outputs/*` (0 B)
- `Data/00_shared/tabular_data/bd_soil_data.csv` (8.8 MB)
- `Data/00_shared/vector_data/Bangladesh/bd_arsenic_data.csv` (39.6 KB)
- `Data/00_shared/vector_data/Bangladesh/Shapefiles/bd_union_GCS.shp` (118.8 MB)
- `Data/00_shared/vector_data/Bangladesh/Shapefiles/bd_upzila_GCS.shp` (43.5 MB)
- download: <https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}>
- download: <https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}>

**07_05_serving_maps.ipynb**

- `Data/00_shared/data_spatial_stat/CA_pm25_2025.csv` (4.1 MB)
- `Data/00_shared/data_web_cloud_gis/08_01_outputs/ndvi_syracuse_20250723.tif` (8.5 MB)
- `Data/00_shared/data_web_cloud_gis/08_01_outputs/stack_deflate_2.tif` (15.2 MB)
- `Data/00_shared/data_web_cloud_gis/08_03_outputs/districts.gpkg` (16.4 MB)
- `Data/00_shared/data_web_cloud_gis/08_05_outputs/*` (0 B)
- `Data/00_shared/data_web_cloud_gis/08_05_outputs/from_api.gpkg` (656.0 KB)
- `Data/00_shared/data_web_cloud_gis/08_05_outputs/unions.pmtiles` (7.6 MB)
- `Data/00_shared/data_web_cloud_gis/s2_syracuse_2025/S2C_18TVN_20250723_0_L2A/stack.tif` (11.6 MB) *(by file name)*
- `Data/00_shared/vector_data/Bangladesh/bd_arsenic_data.csv` (39.6 KB)
- `Data/00_shared/vector_data/Bangladesh/Shapefiles/bd_union_GCS.shp` (118.8 MB)
- `Data/00_shared/vector_data/USA/NY_County_GCS.shp` (151.2 KB)
- download: <https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/18/T/VN/2025/7/S2C_18TVN_20250723_0_L2A/TCI.tif>
- download: <https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.css>
- download: <https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.js>
- download: <https://unpkg.com/pmtiles@3.2.1/dist/pmtiles.js>

**07_06_geospatial_apis.ipynb**

- `Data/00_shared/data_remote_sensing/MODIS/MODIS_NDVI_BD/ndvi_*.tif` (0 B)
- `Data/00_shared/data_web_cloud_gis/08_03_outputs/districts.gpkg` (16.4 MB)
- `Data/00_shared/data_web_cloud_gis/08_06_outputs/downloaded_trend_*.tif` (0 B)
- `Data/00_shared/data_web_cloud_gis/08_06_outputs/trend_*.tif` (0 B)
- `Data/00_shared/tabular_data/bd_soil_data.csv` (8.8 MB)
- `Data/00_shared/vector_data/Bangladesh/bd_arsenic_data.csv` (39.6 KB)
- `Data/00_shared/vector_data/Bangladesh/Shapefiles/bd_union_GCS.shp` (118.8 MB)
- download: <https://unpkg.com/leaflet@1.9.4/dist/leaflet.css>
- download: <https://unpkg.com/leaflet@1.9.4/dist/leaflet.js>
- download: <https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}>

**07_07_web_apps.ipynb**

- `Data/00_shared/data_remote_sensing/MODIS/MODIS_NDVI_BD/ndvi_*.tif` (0 B)
- `Data/00_shared/data_web_cloud_gis/08_03_outputs/districts.gpkg` (16.4 MB)
- `Data/00_shared/data_web_cloud_gis/08_07_outputs/*.parquet` (0 B)
- `Data/00_shared/data_web_cloud_gis/08_07_outputs/district_ndvi.parquet` (93.2 KB)
- `Data/00_shared/data_web_cloud_gis/08_07_outputs/districts.parquet` (188.4 KB)

**07_08_scalable_processing.ipynb**

- `Data/00_shared/data_remote_sensing/Commercial_images/Planetscops/20170620_134925_1_0c41/20170620_134925_1_0c41_metadata.json` (1.9 KB) *(by glob pattern)*
- `Data/00_shared/data_remote_sensing/Commercial_images/SkySat/20170904_154355_ssc2d2_0018/20170904_154355_ssc2d2_0018_metadata.json` (1.5 KB) *(by glob pattern)*
- `Data/00_shared/data_remote_sensing/Data_Ithaca/HLS_Sentinel-2/HLS.S30.T18TUN.2025246T155701.v2.0/item.json` (15.8 KB) *(by glob pattern)*
- `Data/00_shared/data_remote_sensing/Data_Ithaca/Landsat_9/LC09_L2SP_016030_20260814_02_T1/item.json` (36.1 KB) *(by glob pattern)*
- `Data/00_shared/data_remote_sensing/Data_Ithaca/MODIS/MOD13Q1.A2026225.h12v04.061.2026243093237/item.json` (13.9 KB) *(by glob pattern)*
- `Data/00_shared/data_remote_sensing/Data_Ithaca/NAIP/naip_2026-08-17_2026-10-16/item.json` (4.0 KB) *(by glob pattern)*
- `Data/00_shared/data_remote_sensing/MODIS/MODIStsp_2023-08-23.json` (1.2 KB) *(by glob pattern)*
- `Data/00_shared/data_remote_sensing/MODIS/bd_modis_download.json` (1.2 KB) *(by glob pattern)*
- `Data/00_shared/data_remote_sensing/MODIS/bd_modis_download_01.json` (1.2 KB) *(by glob pattern)*
- `Data/00_shared/data_remote_sensing/MODIS/bd_modis_download_GCS.json` (1.2 KB) *(by glob pattern)*
- `Data/00_shared/data_remote_sensing/MODIS/MODIS_NDVI_BD/ndvi_*.tif` (0 B)
- `Data/00_shared/data_remote_sensing/MODIS/MODIStsp/MODIStsp_2023-08-28.json` (1.2 KB) *(by glob pattern)*
- `Data/00_shared/data_remote_sensing/MODIS/MODIStsp/Data/MODIStsp_2023-08-24.json` (1.2 KB) *(by glob pattern)*
- `Data/00_shared/data_remote_sensing/landsat9_vegas_20240708/request.json` (563 B) *(by glob pattern)*
- `Data/00_shared/data_remote_sensing/landsat9_vegas_20240708/scenes.json` (189 B) *(by glob pattern)*
- `Data/00_shared/data_remote_sensing/landsat9_vegas_20240708/LC09_L2SP_039035_20240708_02_T1/item.json` (36.1 KB) *(by glob pattern)*
- `Data/00_shared/data_web_cloud_gis/08_01_outputs/syracuse_s2_2025.zarr` (72.5 MB) *(by glob pattern)*
- `Data/00_shared/data_web_cloud_gis/08_03_outputs/districts.gpkg` (16.4 MB)
- `Data/00_shared/data_web_cloud_gis/08_08_outputs/ndvi_anomaly_trend.tif` (527.3 KB)
- `Data/00_shared/data_web_cloud_gis/08_11_outputs/water/`: 12 files (.json), 1.9 KB
- download: <https://earth-search.aws.element84.com/v1>

**07_09_cloud_platforms.ipynb**

- `Data/00_shared/data_web_cloud_gis/08_03_outputs/districts.gpkg` (16.4 MB)
- `Data/00_shared/data_web_cloud_gis/08_09_outputs/bogra_ndvi_median_2024q1.tif` (4.7 MB)
- `Data/00_shared/data_web_cloud_gis/08_09_outputs/bogra_ndvi_median_2024q1_cog.tif` (6.1 MB)
- `Data/00_shared/data_web_cloud_gis/08_09_outputs/openeo_ndvi_median.json` (3.2 KB)
- download: <https://planetarycomputer.microsoft.com/api/stac/v1>
- download: <https://earth-search.aws.element84.com/v1>
- download: <https://stac.dataspace.copernicus.eu/v1>
- download: <https://openeo.dataspace.copernicus.eu>

**07_10_devops_geospatial.ipynb**

- `Data/00_shared/data_web_cloud_gis/08_10_outputs/monitor_data/observations.parquet` (6.9 KB)

**07_11_capstone.ipynb**

- `Data/00_shared/data_web_cloud_gis/08_10_outputs/monitor_data/observations.parquet` (6.9 KB)
- `Data/00_shared/data_web_cloud_gis/08_11_outputs/*` (0 B)
- `Data/00_shared/data_web_cloud_gis/08_11_outputs/flood_frequency.tif` (1.2 MB)
- `Data/00_shared/data_web_cloud_gis/08_11_outputs/union_series.parquet` (10.5 KB)
- `Data/00_shared/data_web_cloud_gis/08_11_outputs/unions_flood.parquet` (898.4 KB)
- `Data/00_shared/data_web_cloud_gis/08_11_outputs/stac/haor-water/water-2024-01-01/water-2024-01-01.json` (1.3 KB) *(by glob pattern)*
- `Data/00_shared/data_web_cloud_gis/08_11_outputs/stac/haor-water/water-2024-01-13/water-2024-01-13.json` (1.3 KB) *(by glob pattern)*
- `Data/00_shared/data_web_cloud_gis/08_11_outputs/stac/haor-water/water-2024-01-25/water-2024-01-25.json` (1.3 KB) *(by glob pattern)*
- `Data/00_shared/data_web_cloud_gis/08_11_outputs/stac/haor-water/water-2024-02-06/water-2024-02-06.json` (1.2 KB) *(by glob pattern)*
- `Data/00_shared/data_web_cloud_gis/08_11_outputs/stac/haor-water/water-2024-02-18/water-2024-02-18.json` (1.3 KB) *(by glob pattern)*
- `Data/00_shared/data_web_cloud_gis/08_11_outputs/stac/haor-water/water-2024-03-13/water-2024-03-13.json` (1.3 KB) *(by glob pattern)*
- `Data/00_shared/data_web_cloud_gis/08_11_outputs/stac/haor-water/water-2024-03-25/water-2024-03-25.json` (1.3 KB) *(by glob pattern)*
- `Data/00_shared/data_web_cloud_gis/08_11_outputs/stac/haor-water/water-2024-04-06/water-2024-04-06.json` (1.3 KB) *(by glob pattern)*
- `Data/00_shared/data_web_cloud_gis/08_11_outputs/stac/haor-water/water-2024-04-18/water-2024-04-18.json` (1.3 KB) *(by glob pattern)*
- `Data/00_shared/data_web_cloud_gis/08_11_outputs/stac/haor-water/water-2024-05-12/water-2024-05-12.json` (1.3 KB) *(by glob pattern)*
- `Data/00_shared/data_web_cloud_gis/08_11_outputs/stac/haor-water/water-2024-05-24/water-2024-05-24.json` (1.3 KB) *(by glob pattern)*
- `Data/00_shared/data_web_cloud_gis/08_11_outputs/stac/haor-water/water-2024-06-05/water-2024-06-05.json` (1.3 KB) *(by glob pattern)*
- `Data/00_shared/data_web_cloud_gis/08_11_outputs/stac/haor-water/water-2024-06-17/water-2024-06-17.json` (1.3 KB) *(by glob pattern)*
- `Data/00_shared/data_web_cloud_gis/08_11_outputs/stac/haor-water/water-2024-06-29/water-2024-06-29.json` (1.3 KB) *(by glob pattern)*
- `Data/00_shared/data_web_cloud_gis/08_11_outputs/stac/haor-water/water-2024-07-11/water-2024-07-11.json` (1.3 KB) *(by glob pattern)*
- `Data/00_shared/data_web_cloud_gis/08_11_outputs/stac/haor-water/water-2024-07-23/water-2024-07-23.json` (1.3 KB) *(by glob pattern)*
- `Data/00_shared/data_web_cloud_gis/08_11_outputs/stac/haor-water/water-2024-08-04/water-2024-08-04.json` (1.3 KB) *(by glob pattern)*
- `Data/00_shared/data_web_cloud_gis/08_11_outputs/stac/haor-water/water-2024-08-16/water-2024-08-16.json` (1.3 KB) *(by glob pattern)*
- `Data/00_shared/data_web_cloud_gis/08_11_outputs/stac/haor-water/water-2024-08-28/water-2024-08-28.json` (1.2 KB) *(by glob pattern)*
- `Data/00_shared/data_web_cloud_gis/08_11_outputs/stac/haor-water/water-2024-09-09/water-2024-09-09.json` (1.3 KB) *(by glob pattern)*
- `Data/00_shared/data_web_cloud_gis/08_11_outputs/stac/haor-water/water-2024-09-21/water-2024-09-21.json` (1.3 KB) *(by glob pattern)*
- `Data/00_shared/data_web_cloud_gis/08_11_outputs/stac/haor-water/water-2024-10-03/water-2024-10-03.json` (1.3 KB) *(by glob pattern)*
- `Data/00_shared/data_web_cloud_gis/08_11_outputs/stac/haor-water/water-2024-10-15/water-2024-10-15.json` (1.3 KB) *(by glob pattern)*
- `Data/00_shared/data_web_cloud_gis/08_11_outputs/stac/haor-water/water-2024-10-27/water-2024-10-27.json` (1.3 KB) *(by glob pattern)*
- `Data/00_shared/data_web_cloud_gis/08_11_outputs/stac/haor-water/water-2024-11-08/water-2024-11-08.json` (1.3 KB) *(by glob pattern)*
- `Data/00_shared/data_web_cloud_gis/08_11_outputs/stac/haor-water/water-2024-11-20/water-2024-11-20.json` (1.3 KB) *(by glob pattern)*
- `Data/00_shared/data_web_cloud_gis/08_11_outputs/stac/haor-water/water-2024-12-02/water-2024-12-02.json` (1.3 KB) *(by glob pattern)*
- `Data/00_shared/data_web_cloud_gis/08_11_outputs/stac/haor-water/water-2024-12-14/water-2024-12-14.json` (1.3 KB) *(by glob pattern)*
- `Data/00_shared/data_web_cloud_gis/08_11_outputs/stac/haor-water/water-2024-12-26/water-2024-12-26.json` (1.3 KB) *(by glob pattern)*
- `Data/00_shared/data_web_cloud_gis/08_11_outputs/water/*.tif` (0 B)
- `Data/00_shared/data_web_cloud_gis/08_11_outputs/water/water_*.tif` (0 B)
- `Data/00_shared/vector_data/Bangladesh/Shapefiles/bd_union_GCS.shp` (118.8 MB)
- download: <https://planetarycomputer.microsoft.com/api/stac/v1>
- download: <https://unpkg.com/leaflet@1.9.4/dist/leaflet.css>
- download: <https://unpkg.com/leaflet@1.9.4/dist/leaflet.js>
- download: <https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}>

## 08_remote_sensing · Remote Sensing

**Dataset folders used in this chapter**

| Dataset folder | Notebooks | Files | Size |
|---|---:|---:|---:|
| `Data/08_remote_sensing/08_09_image_classification/Corn_belt` | 20 | 31 | 19.8 GB |
| `Data/00_shared/data_remote_sensing` | 12 | 35 | 5.1 GB |
| `Data/08_remote_sensing/08_05_HyperSpectral/Tanger` | 12 | 29 | 57.1 GB |
| `Data/08_remote_sensing/08_02_optical/bd_delta_50yr` | 10 | 4 | 20.9 MB |
| `Data/08_remote_sensing/08_04_uav/infrastructure-free-uav-phenotyping` | 8 | 4 | 0 B |
| `Data/08_remote_sensing/08_06_lidar` | 7 | 6 | 228.9 MB |
| `Data/08_remote_sensing/08_03_radar` | 7 | 5 | 2.4 GB |
| `Data/00_shared/vector_data` | 5 | 2 | 2.3 MB |
| `Data/08_remote_sensing/08_05_HyperSpectral/Pavia` | 5 | 2 | 762.4 MB |
| `Data/08_remote_sensing/08_02_optical` | 5 | 4 | 160.1 MB |
| `Data/08_remote_sensing/08_09_image_classification/Sentinel_2_UB` | 5 | 4 | 9.9 MB |
| `Data/08_remote_sensing/08_03_radar/dinsar_mexico_city` | 4 | 7 | 52.1 MB |
| `Data/08_remote_sensing/08_03_radar/NISAR_bd` | 4 | 5 | 53.8 GB |
| `Data/08_remote_sensing/08_03_radar/RS2_OK153952_PK1408399_DK1372518_FQ17_20150626_052324_HH_VV_HV_VH_SLC` | 4 | 3 | 155.9 KB |
| `Data/08_remote_sensing/08_03_radar/RCM1_OK2801594_PK2803067_1_SCLND_20231025_120938_VV_VH_GRD` | 4 | 5 | 4.9 MB |
| `.` | 3 | 1 | 0 B |
| `Data/08_remote_sensing/08_06_lidar/Ithaca` | 3 | 10 | 9.6 GB |
| `Data/08_remote_sensing/08_08_image_super_resolution/Sentinel_2` | 3 | 2 | 15.1 MB |
| `Data/08_remote_sensing/08_08_image_super_resolution/Landsat8_20190709` | 3 | 3 | 1.4 MB |
| `Data/00_shared/data_geostatistics` | 3 | 2 | 2.2 MB |
| `Data/08_remote_sensing/08_06_lidar/UB` | 3 | 6 | 140.4 MB |
| `Data/08_remote_sensing/08_02_optical/s2_rohingya_2016_2018` | 3 | 1 | 3.3 KB |
| `Data/08_remote_sensing/08_03_radar/s1_rohingya_2017` | 3 | 7 | 43.6 MB |
| `Data/08_remote_sensing/08_08_image_super_resolution/Planet` | 3 | 4 | 75.4 MB |
| `Data/08_remote_sensing/08_09_image_classification/Image_processing` | 3 | 5 | 367.4 MB |
| `Data/08_remote_sensing/08_09_image_classification/Landcover_data` | 3 | 2 | 112.6 KB |
| `Data/08_remote_sensing/08_09_image_classification/Edge_Detection` | 3 | 2 | 17.6 MB |
| `Data/08_remote_sensing/08_09_image_classification/WorldView2_BD` | 3 | 4 | 28.9 MB |
| `Data/08_remote_sensing/08_11_Capstones/bug_fire_sentinel2` | 2 | 23 | 1.5 GB |
| `Data/08_remote_sensing/08_10_object_detection/cv2-pool-detection-master` | 2 | 30 | 22.9 MB |
| `Data/08_remote_sensing/08_11_Capstones` | 2 | 1 | 4.1 GB |
| `Data/08_remote_sensing/08_08_image_super_resolution/HR` | 2 | 1 | 686.3 MB |
| `Data/08_remote_sensing/08_10_object_detection/Edge_Detection` | 2 | 2 | 17.6 MB |
| `Data/08_remote_sensing/08_03_radar/rcm_insar_montreal` | 2 | 5 | 310.0 MB |
| `Data/08_remote_sensing/08_05_HyperSpectral` | 2 | 3 | 1.4 GB |
| `Data/08_remote_sensing/08_03_radar/cop_dem_glo30` | 2 | 2 | 16.6 MB |
| `Data/08_remote_sensing/08_03_radar/S1A_IW_GRDH_1SDV_20240113T115602_20240113T115627_052088_064BA4.SAFE` | 2 | 7 | 0 B |
| `Data/08_remote_sensing/08_11_Capstones/hormuz_strait_sentinel1` | 2 | 36 | 14.7 GB |
| `Data/00_shared/data_web_cloud_gis` | 1 | 1 | 123.8 MB |
| `Data/00_shared/raster_data` | 1 | 1 | 655.7 KB |
| `Data/03_spatial_analysis/Great_plain` | 1 | 2 | 105.5 MB |
| `Data/08_remote_sensing/08_05_HyperSpectral/10m_OH2_Hyperspectral_Sample` | 1 | 1 | 48.8 MB |
| `Data/08_remote_sensing/08_04_uav/Canola dataset` | 1 | 2 | 3.4 MB |
| `Data/08_remote_sensing/08_03_radar/insar_death_valley` | 1 | 11 | 42.2 MB |
| `Data/08_remote_sensing/08_03_radar/RS2_OK153952_PK1408404_DK1372523_F21F_20231123_052042_HH_SGF` | 1 | 1 | 69.6 KB |
| `Data/08_remote_sensing/08_03_radar/RS2_OK153952_PK1408405_DK1372524_W3_20231130_051625_HH_HV_SGF` | 1 | 1 | 72.9 KB |
| `Data/08_remote_sensing/08_03_radar/dualpol_coxsbazar` | 1 | 4 | 0 B |
| `Data/08_remote_sensing/08_02_optical/bangladesh_ndwi_2020.zarr` | 1 | 1 | 0 B |
| `Data/08_remote_sensing/08_02_optical/s3_slstr_vegas` | 1 | 3 | 14.0 MB |
| `Data/08_remote_sensing/08_02_optical/s3_frp_canada_2023` | 1 | 1 | 742.1 KB |
| `Data/08_remote_sensing/08_02_optical/s3_erie_2023` | 1 | 2 | 65.1 MB |
| `Data/08_remote_sensing/08_02_optical/modis_2024` | 1 | 1 | 23.2 KB |
| `Data/08_remote_sensing/08_03_radar/RS2_OK153952_PK1408400_DK1372519_XF0W2_20210705_052511_HH_SLC` | 1 | 1 | 396.7 KB |
| `Data/08_remote_sensing/08_03_radar/RS2_OK153952_PK1408401_DK1372520_MF4N_20231113_051222_HH_SGF` | 1 | 1 | 72.6 KB |
| `Data/08_remote_sensing/08_03_radar/RS2_OK153952_PK1408402_DK1372521_S3_20231116_052449_HH_HV_SGF` | 1 | 1 | 72.9 KB |
| `Data/08_remote_sensing/08_03_radar/RS2_OK153952_PK1408403_DK1372522_SQ1_20231119_053723_HH_VV_HV_VH_SLC` | 1 | 1 | 88.0 KB |
| `Data/08_remote_sensing/08_02_optical/modis13_bangladesh_2020` | 1 | 2 | 1.5 MB |
| `Data/08_remote_sensing/08_04_uav/UAV_BD` | 1 | 4 | 9.2 MB |
| `Data/08_remote_sensing/08_09_image_classification/Image_Fusion` | 1 | 36 | 419.4 MB |
| `Data/08_remote_sensing/08_09_image_classification/Sentinel_01_UB` | 1 | 1 | 559.7 KB |
| `Data/08_remote_sensing/08_09_image_classification/S1_Rice_Bnagladesh` | 1 | 4 | 25.3 MB |
| `Data/08_remote_sensing/08_11_Capstones/NISAR_crop_classification` | 1 | 26 | 13.9 GB |
| `Data/08_remote_sensing/08_10_object_detection/Tree_Detection` | 1 | 2 | 0 B |
| `Data/08_remote_sensing/08_10_object_detection/Corn_belt` | 1 | 4 | 11.3 MB |
| `Data/08_remote_sensing/08_11_Capstones/S1_Rice_Bnagladesh` | 1 | 4 | 1.0 GB |
| `Data/08_remote_sensing/08_11_Capstones/RADARSAT_ICE` | 1 | 3 | 1.6 MB |
| `Data/08_remote_sensing/08_11_Capstones/Sylhet_sentinel1` | 1 | 7 | 56.4 MB |
| `Data/08_remote_sensing/08_11_Capstones/SAR_BD_Coastal_District` | 1 | 4 | 59.0 MB |
| `Data/08_remote_sensing/08_11_Capstones/bd_delta_annual` | 1 | 1 | 0 B |
| `Data/08_remote_sensing/08_11_Capstones/maize_uav_htp` | 1 | 53 | 72.2 MB |
| `Data/08_remote_sensing/08_11_Capstones/s1_BD_charland` | 1 | 9 | 11.7 MB |
| `Data/08_remote_sensing/08_11_Capstones/s1_Big _Sur_wildfires` | 1 | 63 | 2.0 GB |

### 08_01_foundations

**08_01_00_remote_sensing_fundamentals.ipynb**

- `Data/00_shared/data_remote_sensing/LC09_L2SP_138043_20221216_20221223_02_T1/*_SR_B*.TIF` (0 B)
- `Data/00_shared/data_remote_sensing/Landsat8_RAW/*_B*.TIF` (0 B)
- `Data/00_shared/data_remote_sensing/Landsat8_RAW/*_MTL.txt` (0 B)
- `Data/00_shared/data_remote_sensing/Sentinel_2/*.tif` (0 B)
- `Data/00_shared/data_remote_sensing/Sentinel_2/GroundTruth_data.shp` (15.2 KB)
- `Data/00_shared/data_remote_sensing/Sentinel_2/Landuse_ID.csv` (134 B)

**08_01_01_python_geospatial_environment.ipynb**

- `Data/00_shared/data_remote_sensing/LC08_20150706_subset/DEM_TILES/Onondaga_DEM*.TIF` (0 B)
- `Data/00_shared/data_remote_sensing/LC09_L2SP_138043_20221216_20221223_02_T1/LC09_L2SP_138043_20221216_20221223_02_T1_SR_B5.TIF` (60.5 MB)

**08_01_02_raster_vector_essentials.ipynb**

- `Data/00_shared/data_remote_sensing/LC08_20150706_subset/LC08_L1TP_015030_20150716_20170226_01_T1_MTL.txt` (8.5 KB)
- `Data/00_shared/data_remote_sensing/LC08_20150706_subset/multi_band.tif` (24.2 MB)
- `Data/00_shared/data_remote_sensing/LC08_20150706_subset/DEM_TILES/Onondaga_DEM*.TIF` (0 B)
- `Data/00_shared/data_remote_sensing/Sentinel_2/*.tif` (0 B)
- `Data/00_shared/data_remote_sensing/Sentinel_2/B4.tif` (83.9 KB)
- `Data/00_shared/raster_data/USA/Onondaga_DEM.tif` (655.7 KB)
- `Data/00_shared/vector_data/USA/NY_County_GCS.shp` (151.2 KB)

**08_01_03_data_discovery_access.ipynb**

- `Data/00_shared/data_remote_sensing/LC08_20150706_subset/LC08_L1TP_015030_20150716_20170226_01_T1_MTL.txt` (8.5 KB)
- `Data/00_shared/data_remote_sensing/LC08_20150706_subset/multi_band.tif` (24.2 MB)
- `Data/00_shared/data_remote_sensing/LC08_20150706_subset/DEM_TILES/Onondaga_DEM*.TIF` (0 B)
- `Data/00_shared/vector_data/USA/NY_County_GCS.shp` (151.2 KB)

**08_01_04_get_access_rs_images.ipynb**

- `Data/00_shared/data_remote_sensing/Data_Ithaca/Lidar/*.laz` (0 B)
- `Data/00_shared/data_remote_sensing/Data_Ithaca/Sentinel_2/aoi.geojson` (214 B)
- `Data/00_shared/data_remote_sensing/UAV/*.tif` (0 B)
- `Data/00_shared/vector_data/USA/NY_County_GCS.shp` (151.2 KB)
- download: <https://earth-search.aws.element84.com/v1>

### 09_02_optical

**09_02_01_introduction_to_optical_satellite_images.ipynb**

- `Data/00_shared/data_remote_sensing/Commercial_images/GeoEye/053413654010_01/053413654010_01_P001_MUL/13JUL31084528-M2AS-053413654010_01_P001.TIF` (103.8 MB)
- `Data/00_shared/data_remote_sensing/Commercial_images/IKONOS/po_1462947_0000000/po_1462947_bgrn_0000000.tif` (38.1 MB)
- `Data/00_shared/data_remote_sensing/Commercial_images/Jilin/Static_1m_GF03B_Washington_DC/JL1GF03B04_PMS_20210321224611_200044694_102_0007_001_L1_MSS/JL1GF03B04_PMS_20210321224611_200044694_102_0007_001_L1_MSS.tif` (17.8 MB)
- `Data/00_shared/data_remote_sensing/Commercial_images/KOMPSAT/KOMPSAT3_SanFrancisco_7_13_2013_Bundle/K3_20130713205707_06169_16811282_L1G_*.tif` (0 B)
- `Data/00_shared/data_remote_sensing/Commercial_images/KazEOSat/KM002a57MI_003_MUL_L1T.tif` (2.3 GB)
- `Data/00_shared/data_remote_sensing/Commercial_images/KazEOSat/HR_1806010947429/DZZ-HR_20171122023359_038038_E115S31_2A_MS_HR_1806010947429/DZZ-HR_20171122023359_038038_E115S31_2A_MS_HR_1806010947429/imagery.tif` (300.0 MB)
- `Data/00_shared/data_remote_sensing/Commercial_images/Planetscops/20170620_134925_1_0c41/20170620_134925_1_0c41_3B_AnalyticMS.tif` (96.3 MB)
- `Data/00_shared/data_remote_sensing/Commercial_images/Quickbird/053398467010_01/053398467010_01_P001_MUL/13SEP18045419-M2AS-053398467010_01_P001.TIF` (67.8 MB)
- `Data/00_shared/data_remote_sensing/Commercial_images/SPOT6_FerrariWorld_Bundle_9_16_2013/PROD_SPOT6_001/VOL_SPOT6_001_A/IMG_SPOT6_MS_001_A/IMG_SPOT6_MS_201309160638512_SEN_818325101_R1C1.TIF` (9.8 MB)
- `Data/00_shared/data_remote_sensing/Commercial_images/SkySat/20170904_154355_ssc2d2_0018/20170904_154355_ssc2d2_0018_basic_analytic_dn.tif` (2.8 MB)
- `Data/00_shared/data_remote_sensing/Commercial_images/SuperView/SuperView_Sample_Denver_CO/1109180216001_01/SV1-02_20170804_L2A0000463770_1109180216001_01/SV1-02_20170804_L2A0000463770_1109180216001_01-MUX.tiff` (18.9 MB)
- `Data/00_shared/data_remote_sensing/Commercial_images/World_View/052987795060_01/052987795060_01_P001_PAN/10JUN22181146-P2AS-052987795060_01_P001.TIF` (191.9 MB)
- `Data/00_shared/data_remote_sensing/Commercial_images/World_View/056965205010_01/056965205010_01_P001_MUL/17APR18154116-M2AS_R1C1-056965205010_01_P001.TIF` (127.3 MB)
- `Data/00_shared/data_remote_sensing/Commercial_images/World_View/Bakersfield_CA_8band_LG02/050246255010_01/050246255010_01_P001_MUL/24SEP22184516-M2AS-050246255010_01_P001.TIF` (278.2 MB)
- `Data/00_shared/data_remote_sensing/Commercial_images/World_View/DG_MARCH/DG_MAR.tif` (114.2 MB)
- `Data/00_shared/data_remote_sensing/Commercial_images/World_View/DG_MARCH/Label_MARCH.tif` (5.6 MB)
- `Data/00_shared/data_remote_sensing/Commercial_images/World_View/DG_MARCH/Label_March.shp` (46.9 KB)
- `Data/00_shared/data_remote_sensing/Commercial_images/World_View/Namibia_WV3_SWIR_MS_Bundle_Sample/054265823020_01/054265823020_01_P001_MUL/15MAR16092139-M2AS-054265823020_01_P001.TIF` (1.2 GB)
- `Data/00_shared/data_remote_sensing/Commercial_images/World_View/WV2_8band_Bundle_Sydney_Australia/052783824080_01_P001_MUL/11APR03001446-M2AS_R1C1-052783824080_01_P001.TIF` (64.0 MB)
- `Data/08_remote_sensing/08_02_optical/bd_delta_50yr/LC08_L2SP_136045_20180129_02_T1/item.json` (36.0 KB) *(by file name)*
- `Data/08_remote_sensing/08_02_optical/bd_delta_50yr/LC08_L2SP_136045_20180129_02_T1/nir.tif` (2.0 MB) *(by file name)*
- `Data/08_remote_sensing/08_02_optical/bd_delta_50yr/LC08_L2SP_136045_20180129_02_T1/red.tif` (1.8 MB) *(by file name)*
- download: <https://apollomapping.com/download-free-poster>

**09_02_02_preprocessing_of_optical_images.ipynb**

- `Data/08_remote_sensing/08_02_optical/ecostress_lst.tif` (3.4 MB) *(by file name)*
- `Data/08_remote_sensing/08_03_radar/dem_90m.tif` (17.1 MB) *(by file name)*
- `Data/08_remote_sensing/08_03_radar/landsat.tif` (9.7 MB) *(by file name)*
- download: <https://github.com/opengeos/datasets/releases/download>
- download: <https://raw.githubusercontent.com>

**09_02_03_modis_global_monitoring_every_day.ipynb**

- `Data/00_shared/data_geostatistics/BD_Banladesh_BUTM.shp` (2.2 MB)
- `Data/08_remote_sensing/08_02_optical/*.tif` (0 B)
- `Data/08_remote_sensing/08_02_optical/bangladesh_ndwi_2020.zarr` (146.6 MB)
- `Data/08_remote_sensing/08_02_optical/ecostress_lst.tif` (3.4 MB) *(by file name)*
- `Data/08_remote_sensing/08_02_optical/bangladesh_ndwi_2020.zarr/*` (0 B)
- `Data/08_remote_sensing/08_02_optical/bd_delta_50yr/LC08_L2SP_136045_20180129_02_T1/item.json` (36.0 KB) *(by file name)*
- `Data/08_remote_sensing/08_02_optical/bd_delta_50yr/LC08_L2SP_136045_20180129_02_T1/nir.tif` (2.0 MB) *(by file name)*
- `Data/08_remote_sensing/08_02_optical/bd_delta_50yr/LC08_L2SP_136045_20180129_02_T1/red.tif` (1.8 MB) *(by file name)*
- `Data/08_remote_sensing/08_02_optical/modis13_bangladesh_2020/MOD13Q1.A2020001.h25v06.061.2020326051742/mir.tif` (1.4 MB) *(by file name)*
- `Data/08_remote_sensing/08_02_optical/modis13_bangladesh_2020/MOD13Q1.A2020001.h25v06.061.2020326051742/quality.tif` (112.0 KB) *(by file name)*
- `Data/08_remote_sensing/08_02_optical/modis_2024/MOD09Q1.A2024001.h08v05.061.2024011194111/state.tif` (23.2 KB) *(by file name)*
- `Data/08_remote_sensing/08_03_radar/dem_90m.tif` (17.1 MB) *(by file name)*
- `Data/08_remote_sensing/08_03_radar/landsat.tif` (9.7 MB) *(by file name)*
- download: <https://planetarycomputer.microsoft.com/api/stac/v1>
- download: <https://github.com/opengeos/datasets/releases/download>
- download: <https://raw.githubusercontent.com>

**09_02_04_landsat_half_a_century_of_earth_observation.ipynb**

- `Data/08_remote_sensing/08_02_optical/ecostress_lst.tif` (3.4 MB) *(by file name)*
- `Data/08_remote_sensing/08_03_radar/dem_90m.tif` (17.1 MB) *(by file name)*
- download: <https://planetarycomputer.microsoft.com/api/stac/v1>
- download: <https://github.com/opengeos/datasets/releases/download>
- download: <https://raw.githubusercontent.com>

**09_02_05_sentinel_2_red_edge_10_m_and_five_day_revisits.ipynb**

- `Data/00_shared/vector_data/USA/NY_County_GCS.shp` (151.2 KB)
- `Data/08_remote_sensing/08_02_optical/bd_delta_50yr/LC08_L2SP_136045_20180129_02_T1/item.json` (36.0 KB) *(by file name)*
- `Data/08_remote_sensing/08_02_optical/bd_delta_50yr/LC08_L2SP_136045_20180129_02_T1/red.tif` (1.8 MB) *(by file name)*
- `Data/08_remote_sensing/08_02_optical/s2_rohingya_2016_2018/S2A_MSIL2A_20160105T042142_R090_T46QCJ_20210527T085101/scl.tif` (1.1 KB) *(by file name)*
- download: <https://planetarycomputer.microsoft.com/api/stac/v1>
- download: <https://github.com/opengeos/datasets/releases/download>

**09_02_06_sentinel_3_ocean_colour_surface_temperature_and_altimetry.ipynb**

- `Data/08_remote_sensing/08_02_optical/s3_erie_2023/S3A_OL_2_WFR_20230919T153401_20230919T153701_0180_103_282_2340/geo_coordinates.nc` (57.5 MB) *(by file name)*
- `Data/08_remote_sensing/08_02_optical/s3_erie_2023/S3A_OL_2_WFR_20230919T153401_20230919T153701_0180_103_282_2340/wqsf.nc` (7.6 MB) *(by file name)*
- `Data/08_remote_sensing/08_02_optical/s3_frp_canada_2023/S3A_SL_2_FRP_20230516T042732_20230516T043032_0179_099_019_0720/FRP_in.nc` (742.1 KB) *(by file name)*
- `Data/08_remote_sensing/08_02_optical/s3_slstr_vegas/S3A_SL_2_LST_20230715T050921_20230715T051221_0180_101_105_0540/LST_in.nc` (4.0 MB) *(by file name)*
- `Data/08_remote_sensing/08_02_optical/s3_slstr_vegas/S3A_SL_2_LST_20230715T050921_20230715T051221_0180_101_105_0540/flags_in.nc` (906.8 KB) *(by file name)*
- `Data/08_remote_sensing/08_02_optical/s3_slstr_vegas/S3A_SL_2_LST_20230715T050921_20230715T051221_0180_101_105_0540/geodetic_in.nc` (9.1 MB) *(by file name)*
- download: <https://planetarycomputer.microsoft.com/api/stac/v1>

**09_02_07_sentinel_5p_and_sentinel_5_mapping_air_pollution_from_orbit.ipynb**

- `Data/00_shared/vector_data/Bangladesh/Shapefiles/bd_boundary_GCS.shp` (1.7 MB)
- download: <https://planetarycomputer.microsoft.com/api/stac/v1>

### 09_03_radar

**09_03_00_sar_fundamentals.ipynb**

- `Data/08_remote_sensing/08_03_radar/dem_90m.tif` (17.1 MB) *(by file name)*
- `Data/08_remote_sensing/08_03_radar/landsat.tif` (9.7 MB) *(by file name)*
- `Data/08_remote_sensing/08_03_radar/s1_annotation.xml` (1.7 MB) *(by file name)*
- `Data/08_remote_sensing/08_03_radar/s1_vv_db.tif` (227.8 KB) *(by file name)*
- download: <https://github.com/opengeos/datasets/releases/download>
- download: <https://raw.githubusercontent.com>

**09_03_01_02_sentinel_1_grd_processing.ipynb**

- `Data/08_remote_sensing/08_03_radar/S1A_IW_GRDH_1SDV_20240113T115602_20240113T115627_052088_064BA4.SAFE` (1.2 GB)
- `Data/08_remote_sensing/08_03_radar/S1A_IW_GRDH_1SDV_20240113T115602_20240113T115627_052088_064BA4.SAFE/annotation/s1a-iw-grd-vv-*.xml` (0 B)
- `Data/08_remote_sensing/08_03_radar/S1A_IW_GRDH_1SDV_20240113T115602_20240113T115627_052088_064BA4.SAFE/annotation/calibration/calibration-s1a-iw-grd-vh-*.xml` (0 B)
- `Data/08_remote_sensing/08_03_radar/S1A_IW_GRDH_1SDV_20240113T115602_20240113T115627_052088_064BA4.SAFE/annotation/calibration/calibration-s1a-iw-grd-vv-*.xml` (0 B)
- `Data/08_remote_sensing/08_03_radar/S1A_IW_GRDH_1SDV_20240113T115602_20240113T115627_052088_064BA4.SAFE/annotation/calibration/noise-s1a-iw-grd-vh-*.xml` (0 B)
- `Data/08_remote_sensing/08_03_radar/S1A_IW_GRDH_1SDV_20240113T115602_20240113T115627_052088_064BA4.SAFE/annotation/calibration/noise-s1a-iw-grd-vv-*.xml` (0 B)
- `Data/08_remote_sensing/08_03_radar/S1A_IW_GRDH_1SDV_20240113T115602_20240113T115627_052088_064BA4.SAFE/measurement/s1a-iw-grd-vh-*.tiff` (0 B)
- `Data/08_remote_sensing/08_03_radar/S1A_IW_GRDH_1SDV_20240113T115602_20240113T115627_052088_064BA4.SAFE/measurement/s1a-iw-grd-vv-*.tiff` (0 B)
- `Data/08_remote_sensing/08_03_radar/cop_dem_glo30/*.tif` (0 B)
- `Data/08_remote_sensing/08_03_radar/cop_dem_glo30/coxsbazar_dem.tif` (8.3 MB)
- download: <https://planetarycomputer.microsoft.com/api/stac/v1>

**09_03_01_03_sentinel_1_multitemporal_grd_time_series_deforestation_around_the_rohingya_camps_2017.ipynb**

- `Data/08_remote_sensing/08_02_optical/bd_delta_50yr/LC08_L2SP_136045_20180129_02_T1/item.json` (36.0 KB) *(by file name)*
- `Data/08_remote_sensing/08_03_radar/s1_rohingya_2017/copdem30_stack_grid.nc` (1.1 MB)
- `Data/08_remote_sensing/08_03_radar/s1_rohingya_2017/grd_items_2017.json` (673.4 KB)
- `Data/08_remote_sensing/08_03_radar/s1_rohingya_2017/grd_sigma0_track*_2017.nc` (0 B)
- `Data/08_remote_sensing/08_03_radar/s1_rohingya_2017/rtc_gamma0_track*_2017.nc` (0 B)
- download: <https://planetarycomputer.microsoft.com/api/stac/v1>

**09_03_01_04_topographic_insar_a_dem_from_a_sentinel_1_slc_pair.ipynb**

- `Data/08_remote_sensing/08_03_radar/insar_death_valley/`: 8 files (.json, .parquet, .tif), 42.2 MB
- `Data/08_remote_sensing/08_03_radar/insar_death_valley/slc/*_*.json` (0 B)
- `Data/08_remote_sensing/08_03_radar/insar_death_valley/slc/*_*.npy` (0 B)
- `Data/08_remote_sensing/08_03_radar/insar_death_valley/slc/*_*.xml` (0 B)
- download: <https://planetarycomputer.microsoft.com/api/stac/v1>

**09_03_01_05_differential_insar_dinsar.ipynb**

- `Data/08_remote_sensing/08_03_radar/dinsar_mexico_city/asf_bursts_2025.json` (132.2 KB) *(by file name)*
- `Data/08_remote_sensing/08_03_radar/dinsar_mexico_city/asf_stack_2025.json` (753.2 KB) *(by file name)*
- `Data/08_remote_sensing/08_03_radar/dinsar_mexico_city/copdem30_ellipsoid.tif` (8.2 MB)
- `Data/08_remote_sensing/08_03_radar/dinsar_mexico_city/copdem30_geoid.tif` (4.1 MB)
- `Data/08_remote_sensing/08_03_radar/dinsar_mexico_city/slc/*_*.json` (0 B)
- `Data/08_remote_sensing/08_03_radar/dinsar_mexico_city/slc/*_*.npy` (0 B)
- `Data/08_remote_sensing/08_03_radar/dinsar_mexico_city/slc/*_*.xml` (0 B)
- download: <https://planetarycomputer.microsoft.com/api/stac/v1>
- download: <https://geodesy.unr.edu/gps_timeseries/IGS20/tenv3/IGS20/*.tenv3>
- download: <https://geodesy.unr.edu/gps_timeseries/IGS20/tenv3/IGS20/>

**09_03_01_06_sbas_small_baseline_subset_insar.ipynb**

- `Data/08_remote_sensing/08_03_radar/dinsar_mexico_city/asf_bursts_2025.json` (132.2 KB) *(by file name)*
- `Data/08_remote_sensing/08_03_radar/dinsar_mexico_city/asf_stack_2025.json` (753.2 KB) *(by file name)*
- `Data/08_remote_sensing/08_03_radar/dinsar_mexico_city/copdem30_ellipsoid.tif` (8.2 MB)
- `Data/08_remote_sensing/08_03_radar/dinsar_mexico_city/copdem30_geoid.tif` (4.1 MB)
- `Data/08_remote_sensing/08_03_radar/dinsar_mexico_city/slc/*_*.json` (0 B)
- `Data/08_remote_sensing/08_03_radar/dinsar_mexico_city/slc/*_*.npy` (0 B)
- `Data/08_remote_sensing/08_03_radar/dinsar_mexico_city/slc/*_*.xml` (0 B)
- download: <https://planetarycomputer.microsoft.com/api/stac/v1>
- download: <https://geodesy.unr.edu/gps_timeseries/IGS20/tenv3/IGS20/*.tenv3>
- download: <https://geodesy.unr.edu/gps_timeseries/IGS20/tenv3/IGS20/>

**09_03_01_07_persistent_scatterer_insar_ps_insar.ipynb**

- `Data/08_remote_sensing/08_03_radar/dinsar_mexico_city/asf_bursts_2025.json` (132.2 KB) *(by file name)*
- `Data/08_remote_sensing/08_03_radar/dinsar_mexico_city/asf_stack_2025.json` (753.2 KB) *(by file name)*
- `Data/08_remote_sensing/08_03_radar/dinsar_mexico_city/copdem30_ellipsoid.tif` (8.2 MB)
- `Data/08_remote_sensing/08_03_radar/dinsar_mexico_city/copdem30_geoid.tif` (4.1 MB)
- `Data/08_remote_sensing/08_03_radar/dinsar_mexico_city/slc/*_*.json` (0 B)
- `Data/08_remote_sensing/08_03_radar/dinsar_mexico_city/slc/*_*.npy` (0 B)
- `Data/08_remote_sensing/08_03_radar/dinsar_mexico_city/slc/*_*.xml` (0 B)
- download: <https://planetarycomputer.microsoft.com/api/stac/v1>
- download: <https://geodesy.unr.edu/gps_timeseries/IGS20/tenv3/IGS20/*.tenv3>
- download: <https://geodesy.unr.edu/gps_timeseries/IGS20/tenv3/IGS20/>

**09_03_01_08_dual_pol_polarimetric_decomposition_slc.ipynb**

- `Data/08_remote_sensing/08_03_radar/cop_dem_glo30/coxsbazar_dem.tif` (8.3 MB)
- `Data/08_remote_sensing/08_03_radar/dualpol_coxsbazar/slc/*_*_*.json` (0 B)
- `Data/08_remote_sensing/08_03_radar/dualpol_coxsbazar/slc/*_*_*.npy` (0 B)
- `Data/08_remote_sensing/08_03_radar/dualpol_coxsbazar/slc/*_*_*.xml` (0 B)
- `Data/08_remote_sensing/08_03_radar/dualpol_coxsbazar/slc/*_*_VV.xml` (0 B)

**09_03_01_09_interferometric_coherence_of_slcs.ipynb**

- `Data/08_remote_sensing/08_03_radar/dinsar_mexico_city/copdem30_ellipsoid.tif` (8.2 MB)
- `Data/08_remote_sensing/08_03_radar/dinsar_mexico_city/copdem30_geoid.tif` (4.1 MB)
- `Data/08_remote_sensing/08_03_radar/dinsar_mexico_city/slc/*_*.json` (0 B)
- `Data/08_remote_sensing/08_03_radar/dinsar_mexico_city/slc/*_*.npy` (0 B)
- `Data/08_remote_sensing/08_03_radar/dinsar_mexico_city/slc/*_*.xml` (0 B)

**09_03_01_10_sentinel_1_based_indices.ipynb**

- `Data/08_remote_sensing/08_03_radar/S1A_IW_GRDH_1SDV_20240113T115602_20240113T115627_052088_064BA4.SAFE` (1.2 GB)
- `Data/08_remote_sensing/08_03_radar/RS2_OK153952_PK1408399_DK1372518_FQ17_20150626_052324_HH_VV_HV_VH_SLC/lutSigma.xml` (44.6 KB) *(by file name)*
- `Data/08_remote_sensing/08_03_radar/S1A_IW_GRDH_1SDV_20240113T115602_20240113T115627_052088_064BA4.SAFE/annotation/calibration/calibration-s1a-iw-grd-vh-*.xml` (0 B)
- `Data/08_remote_sensing/08_03_radar/S1A_IW_GRDH_1SDV_20240113T115602_20240113T115627_052088_064BA4.SAFE/annotation/calibration/calibration-s1a-iw-grd-vv-*.xml` (0 B)
- `Data/08_remote_sensing/08_03_radar/S1A_IW_GRDH_1SDV_20240113T115602_20240113T115627_052088_064BA4.SAFE/annotation/calibration/noise-s1a-iw-grd-vh-*.xml` (0 B)
- `Data/08_remote_sensing/08_03_radar/S1A_IW_GRDH_1SDV_20240113T115602_20240113T115627_052088_064BA4.SAFE/annotation/calibration/noise-s1a-iw-grd-vv-*.xml` (0 B)
- `Data/08_remote_sensing/08_03_radar/S1A_IW_GRDH_1SDV_20240113T115602_20240113T115627_052088_064BA4.SAFE/measurement/s1a-iw-grd-vh-*.tiff` (0 B)
- `Data/08_remote_sensing/08_03_radar/S1A_IW_GRDH_1SDV_20240113T115602_20240113T115627_052088_064BA4.SAFE/measurement/s1a-iw-grd-vv-*.tiff` (0 B)
- `Data/08_remote_sensing/08_03_radar/s1_rohingya_2017/grd_sigma0_track77_2017.nc` (20.9 MB)
- `Data/08_remote_sensing/08_03_radar/s1_rohingya_2017/rtc_gamma0_track77_2017.nc` (20.9 MB)

**09_03_02_02_radarsat_grd_data.ipynb**

- `Data/08_remote_sensing/08_03_radar/RCM1_OK2801594_PK2803067_1_SCLND_20231025_120938_VV_VH_GRD/imagery/2803067_1_*.tif` (0 B)
- `Data/08_remote_sensing/08_03_radar/RCM1_OK2801594_PK2803067_1_SCLND_20231025_120938_VV_VH_GRD/metadata/product.xml` (2.4 MB)
- `Data/08_remote_sensing/08_03_radar/RCM1_OK2801594_PK2803067_1_SCLND_20231025_120938_VV_VH_GRD/metadata/calibration/lutSigma_*.xml` (0 B)
- `Data/08_remote_sensing/08_03_radar/RCM1_OK2801594_PK2803067_1_SCLND_20231025_120938_VV_VH_GRD/metadata/calibration/noiseLevels_*.xml` (0 B)

**09_03_02_03_radarsat_constellation_mission_rcm_insar_processing.ipynb**

- `Data/08_remote_sensing/08_03_radar/rcm_insar_montreal/copdem30_ellipsoid.tif` (13.9 MB)
- `Data/08_remote_sensing/08_03_radar/rcm_insar_montreal/RCM1_OK2678697_PK2744956_1_16M11_20230913_224404_VV_SLC/imagery/2744956_1_VV.tif` (147.7 MB)
- `Data/08_remote_sensing/08_03_radar/rcm_insar_montreal/RCM1_OK2678697_PK2744956_1_16M11_20230913_224404_VV_SLC/metadata/product.xml` (167.1 KB)
- `Data/08_remote_sensing/08_03_radar/rcm_insar_montreal/RCM2_OK2678697_PK2750939_1_16M11_20230917_224416_VV_SLC/imagery/2750939_1_VV.tif` (147.7 MB)
- `Data/08_remote_sensing/08_03_radar/rcm_insar_montreal/RCM2_OK2678697_PK2750939_1_16M11_20230917_224416_VV_SLC/metadata/product.xml` (166.2 KB)
- download: <https://planetarycomputer.microsoft.com/api/stac/v1>

**09_03_02_04_radarsat_polarimetric_sar_polsar.ipynb**

- `Data/08_remote_sensing/08_03_radar/RS2_OK153952_PK1408399_DK1372518_FQ17_20150626_052324_HH_VV_HV_VH_SLC/imagery_*.tif` (0 B)

**09_03_02_05_compact_polarimetry_cp.ipynb**

- `Data/08_remote_sensing/08_03_radar/RS2_OK153952_PK1408399_DK1372518_FQ17_20150626_052324_HH_VV_HV_VH_SLC/imagery_*.tif` (0 B)

**09_03_03_02_nisar_rslc_range_doppler_single_look_complex_processing.ipynb**

- `Data/08_remote_sensing/08_03_radar/NISAR_bd/AOI_coastal.geojson` (458 B)
- `Data/08_remote_sensing/08_03_radar/NISAR_bd/NISAR_L1_PR_RSLC_031_069_A_013_4005_DHDH_A_20260922T232058_20260922T232133_P05023_N_F_J_001.h5` (24.7 GB)

**09_03_03_03_nisar_gslc_geocoded_single_look_complex_processing.ipynb**

- `Data/08_remote_sensing/08_03_radar/NISAR_bd/AOI_coastal.geojson` (458 B)
- `Data/08_remote_sensing/08_03_radar/NISAR_bd/NISAR_L2_PR_GSLC_031_069_A_013_4005_DHDH_A_20260922T232058_20260922T232133_P05023_N_F_J_001.h5` (21.0 GB)

**09_03_03_04_nisar_gcov_geocoded_polarimetric_covariance_processing.ipynb**

- `Data/08_remote_sensing/08_03_radar/NISAR_bd/AOI_coastal.geojson` (458 B)
- `Data/08_remote_sensing/08_03_radar/NISAR_bd/NISAR_L2_PR_GCOV_031_069_A_013_4005_DHDH_A_20260922T232058_20260922T232133_P05023_N_F_J_001.h5` (5.9 GB)

**09_03_03_05_nisar_gunw_geocoded_unwrapped_interferogram.ipynb**

- `Data/08_remote_sensing/08_03_radar/NISAR_bd/AOI_coastal.geojson` (458 B)
- `Data/08_remote_sensing/08_03_radar/NISAR_bd/NISAR_L2_PR_GUNW_031_069_A_013_032_4000_SH_20260922T232058_20260922T232133_20261004T232058_20261004T232133_P05023_N_F_J_001.h5` (2.2 GB)

### 09_04_uav

**09_04_01_uav_fundamentals_and_mission_planning.ipynb**

- `Data/08_remote_sensing/08_04_uav/infrastructure-free-uav-phenotyping/2025-08-22/*.png` (0 B)

**09_04_02_reading_and_inspecting_uav_imagery.ipynb**

- `Data/08_remote_sensing/08_04_uav/Canola dataset/canola_06082019/NIR/5.png` (1.7 MB)
- `Data/08_remote_sensing/08_04_uav/Canola dataset/canola_06082019/RGB/5.png` (1.7 MB)
- `Data/08_remote_sensing/08_04_uav/infrastructure-free-uav-phenotyping/2025-08-22/*.png` (0 B)

**09_04_03_photogrammetry_orthomosaic_dsm_and_dtm.ipynb**

- `Data/08_remote_sensing/08_04_uav/infrastructure-free-uav-phenotyping/2025-08-22/*.png` (0 B)

**09_04_04_rgb_vegetation_indices_and_canopy_cover.ipynb**

- `Data/08_remote_sensing/08_04_uav/UAV_BD/multi_bwh_mc_20150212.tif` (3.2 MB)
- `Data/08_remote_sensing/08_04_uav/UAV_BD/multi_bwh_mc_20150227.tif` (2.9 MB)
- `Data/08_remote_sensing/08_04_uav/UAV_BD/multi_bwh_mc_20150312.tif` (3.1 MB)
- `Data/08_remote_sensing/08_04_uav/UAV_BD/plot_bd_wheat.shp` (1.3 KB)
- `Data/08_remote_sensing/08_04_uav/infrastructure-free-uav-phenotyping/2025-08-22/*.png` (0 B)

**09_04_07_plant_counting_and_size.ipynb**

- `Data/08_remote_sensing/08_04_uav/infrastructure-free-uav-phenotyping/2025-08-22/*.png` (0 B)

**09_04_08_crop_health_sam_plant_masks_and_senescence_detection.ipynb**

- `Data/08_remote_sensing/08_04_uav/infrastructure-free-uav-phenotyping/2025-09-12/*.png` (0 B)
- `Data/08_remote_sensing/08_04_uav/infrastructure-free-uav-phenotyping/2025-10-01/*.png` (0 B)
- `Data/08_remote_sensing/08_04_uav/infrastructure-free-uav-phenotyping/2025-10-03/*.png` (0 B)

**09_04_09_panicle_detection_with_yolo.ipynb**

- `*.txt` (0 B)
- `Data/08_remote_sensing/08_04_uav/infrastructure-free-uav-phenotyping/2025-10-03/*.png` (0 B)

**09_04_10_season_long_phenotyping.ipynb**

- `Data/08_remote_sensing/08_04_uav/infrastructure-free-uav-phenotyping/2025-09-12/*.png` (0 B)

### 09_05_HyperSpectral

**09_05_00_introduction_to_imaging_spectroscopy.ipynb**

- `Data/08_remote_sensing/08_05_HyperSpectral/10m_OH2_Hyperspectral_Sample/Spanish-OHS/HAZ2_20180715215912_0002_L1_MSS_CCD1/HAZ2_20180715215912_0002_L1_MSS_B1_CCD1.tif` (48.8 MB) *(by glob pattern)*
- `Data/08_remote_sensing/08_05_HyperSpectral/Pavia/pavia.npy` (152.5 MB)
- `Data/08_remote_sensing/08_05_HyperSpectral/Tanger/scenes.json` (3.5 KB)
- `Data/08_remote_sensing/08_05_HyperSpectral/Tanger/GHG-plumes/20241004_081921_28_4001/20241004_081921_28_4001_ortho_radiance_hdf5.h5` (682.3 MB) *(by glob pattern)*
- `Data/08_remote_sensing/08_05_HyperSpectral/Tanger/GHG-plumes/20241004_081921_28_4001/20241004_081921_28_4001_ortho_sr_hdf5.h5` (1.3 GB) *(by glob pattern)*
- `Data/08_remote_sensing/08_05_HyperSpectral/Tanger/GHG-plumes/20241004_081921_28_4001/20241004_081921_28_4001_thumb.png` (802.3 KB) *(by glob pattern)*
- `Data/08_remote_sensing/08_05_HyperSpectral/Tanger/agriculture/*/*_ortho_sr_hdf5.h5` (0 B)
- `Data/08_remote_sensing/08_05_HyperSpectral/Tanger/agriculture/20250224_145149_32_4001/20250224_145149_32_4001_ortho_radiance_hdf5.h5` (459.6 MB) *(by glob pattern)*
- `Data/08_remote_sensing/08_05_HyperSpectral/Tanger/agriculture/20250224_145149_32_4001/20250224_145149_32_4001_ortho_sr_hdf5.h5` (857.8 MB) *(by glob pattern)*
- `Data/08_remote_sensing/08_05_HyperSpectral/Tanger/agriculture/20250224_145149_32_4001/20250224_145149_32_4001_thumb.png` (564.0 KB) *(by glob pattern)*
- `Data/08_remote_sensing/08_05_HyperSpectral/Tanger/coastal-water-bodies/20250225_153627_16_4001/20250225_153627_16_4001_ortho_radiance_hdf5.h5` (459.9 MB) *(by glob pattern)*
- `Data/08_remote_sensing/08_05_HyperSpectral/Tanger/coastal-water-bodies/20250225_153627_16_4001/20250225_153627_16_4001_ortho_sr_hdf5.h5` (838.0 MB) *(by glob pattern)*
- `Data/08_remote_sensing/08_05_HyperSpectral/Tanger/coastal-water-bodies/20250225_153627_16_4001/20250225_153627_16_4001_thumb.png` (496.1 KB) *(by glob pattern)*
- `Data/08_remote_sensing/08_05_HyperSpectral/Tanger/energy-mining/20250205_175845_00_4001/20250205_175845_00_4001_ortho_radiance_hdf5.h5` (661.6 MB) *(by glob pattern)*
- `Data/08_remote_sensing/08_05_HyperSpectral/Tanger/energy-mining/20250205_175845_00_4001/20250205_175845_00_4001_ortho_sr_hdf5.h5` (1.2 GB) *(by glob pattern)*
- `Data/08_remote_sensing/08_05_HyperSpectral/Tanger/energy-mining/20250205_175845_00_4001/20250205_175845_00_4001_thumb.png` (725.9 KB) *(by glob pattern)*
- `Data/08_remote_sensing/08_05_HyperSpectral/Tanger/fire/20241215_185916_33_4001/20241215_185916_33_4001_ortho_radiance_hdf5.h5` (429.2 MB) *(by glob pattern)*
- `Data/08_remote_sensing/08_05_HyperSpectral/Tanger/fire/20241215_185916_33_4001/20241215_185916_33_4001_ortho_sr_hdf5.h5` (784.8 MB) *(by glob pattern)*
- `Data/08_remote_sensing/08_05_HyperSpectral/Tanger/fire/20241215_185916_33_4001/20241215_185916_33_4001_thumb.png` (487.2 KB) *(by glob pattern)*
- `Data/08_remote_sensing/08_05_HyperSpectral/Tanger/snow-ice/20250411_163421_16_4001/20250411_163421_16_4001_ortho_radiance_hdf5.h5` (653.8 MB) *(by glob pattern)*
- `Data/08_remote_sensing/08_05_HyperSpectral/Tanger/snow-ice/20250411_163421_16_4001/20250411_163421_16_4001_ortho_sr_hdf5.h5` (1.2 GB) *(by glob pattern)*
- `Data/08_remote_sensing/08_05_HyperSpectral/Tanger/snow-ice/20250411_163421_16_4001/20250411_163421_16_4001_thumb.png` (555.1 KB) *(by glob pattern)*
- `Data/08_remote_sensing/08_05_HyperSpectral/Tanger/urban/20250423_193451_57_4001/20250423_193451_57_4001_ortho_radiance_hdf5.h5` (629.5 MB) *(by glob pattern)*
- `Data/08_remote_sensing/08_05_HyperSpectral/Tanger/urban/20250423_193451_57_4001/20250423_193451_57_4001_ortho_sr_hdf5.h5` (1.2 GB) *(by glob pattern)*
- `Data/08_remote_sensing/08_05_HyperSpectral/Tanger/urban/20250423_193451_57_4001/20250423_193451_57_4001_thumb.png` (1.1 MB) *(by glob pattern)*

**09_05_01_reading_and_exploring_hyperspectral_data.ipynb**

- `Data/08_remote_sensing/08_05_HyperSpectral/20250224_145149_32_4001_basic_radiance_hdf5.h5` (432.3 MB)
- `Data/08_remote_sensing/08_05_HyperSpectral/20250516_181930_35_4001_basic_radiance_hdf5.h5` (536.3 MB)
- `Data/08_remote_sensing/08_05_HyperSpectral/20250516_181930_35_4001_geolocation_array.tif` (3.9 MB)
- `Data/08_remote_sensing/08_05_HyperSpectral/Pavia/pavia.npy` (152.5 MB)
- `Data/08_remote_sensing/08_05_HyperSpectral/Tanger/scenes.json` (3.5 KB)
- `Data/08_remote_sensing/08_05_HyperSpectral/Tanger/agriculture/20250224_145149_32_4001/20250224_145149_32_4001_ortho_radiance_hdf5.h5` (459.6 MB) *(by glob pattern)*
- `Data/08_remote_sensing/08_05_HyperSpectral/Tanger/agriculture/20250224_145149_32_4001/20250224_145149_32_4001_ortho_sr_hdf5.h5` (857.8 MB) *(by glob pattern)*
- `Data/08_remote_sensing/08_05_HyperSpectral/Tanger/urban/20250423_193451_57_4001/20250423_193451_57_4001_ortho_radiance_hdf5.h5` (629.5 MB) *(by glob pattern)*
- `Data/08_remote_sensing/08_05_HyperSpectral/Tanger/urban/20250423_193451_57_4001/20250423_193451_57_4001_ortho_sr_hdf5.h5` (1.2 GB) *(by glob pattern)*

**09_05_02_preprocessing_hyperspectral_cubes.ipynb**

- `Data/08_remote_sensing/08_05_HyperSpectral/20250224_145149_32_4001_basic_radiance_hdf5.h5` (432.3 MB)
- `Data/08_remote_sensing/08_05_HyperSpectral/Tanger/agriculture/20250224_145149_32_4001/20250224_145149_32_4001_ortho_radiance_hdf5.h5` (459.6 MB) *(by glob pattern)*
- `Data/08_remote_sensing/08_05_HyperSpectral/Tanger/agriculture/20250224_145149_32_4001/20250224_145149_32_4001_ortho_sr_hdf5.h5` (857.8 MB) *(by glob pattern)*
- `Data/08_remote_sensing/08_05_HyperSpectral/Tanger/energy-mining/20250205_175845_00_4001/20250205_175845_00_4001_ortho_radiance_hdf5.h5` (661.6 MB) *(by glob pattern)*
- `Data/08_remote_sensing/08_05_HyperSpectral/Tanger/energy-mining/20250205_175845_00_4001/20250205_175845_00_4001_ortho_sr_hdf5.h5` (1.2 GB) *(by glob pattern)*
- `Data/08_remote_sensing/08_05_HyperSpectral/Tanger/fire/20241215_185916_33_4001/20241215_185916_33_4001_ortho_radiance_hdf5.h5` (429.2 MB) *(by glob pattern)*
- `Data/08_remote_sensing/08_05_HyperSpectral/Tanger/fire/20241215_185916_33_4001/20241215_185916_33_4001_ortho_sr_hdf5.h5` (784.8 MB) *(by glob pattern)*

**09_05_03_dimensionality_reduction_and_band_selection.ipynb**

- `Data/08_remote_sensing/08_05_HyperSpectral/Pavia/Pavia_gt.mat` (34.1 KB)
- `Data/08_remote_sensing/08_05_HyperSpectral/Pavia/pavia.npy` (152.5 MB)
- `Data/08_remote_sensing/08_05_HyperSpectral/Tanger/urban/20250423_193451_57_4001/20250423_193451_57_4001_ortho_radiance_hdf5.h5` (629.5 MB) *(by glob pattern)*
- `Data/08_remote_sensing/08_05_HyperSpectral/Tanger/urban/20250423_193451_57_4001/20250423_193451_57_4001_ortho_sr_hdf5.h5` (1.2 GB) *(by glob pattern)*

**09_05_04_narrowband_indices_and_vegetation_analysis.ipynb**

- `Data/08_remote_sensing/08_05_HyperSpectral/Tanger/ROCX2025/20250919_170233_04_4001/20250919_170233_04_4001_ortho_radiance_hdf5.h5` (682.5 MB) *(by glob pattern)*
- `Data/08_remote_sensing/08_05_HyperSpectral/Tanger/ROCX2025/20250919_170233_04_4001/20250919_170233_04_4001_ortho_sr_hdf5.h5` (1.2 GB) *(by glob pattern)*
- `Data/08_remote_sensing/08_05_HyperSpectral/Tanger/agriculture/20250224_145149_32_4001/20250224_145149_32_4001_ortho_radiance_hdf5.h5` (459.6 MB) *(by glob pattern)*
- `Data/08_remote_sensing/08_05_HyperSpectral/Tanger/agriculture/20250224_145149_32_4001/20250224_145149_32_4001_ortho_sr_hdf5.h5` (857.8 MB) *(by glob pattern)*
- `Data/08_remote_sensing/08_05_HyperSpectral/Tanger/natural-lands/20250221_102048_16_4001/20250221_102048_16_4001_ortho_radiance_hdf5.h5` (483.0 MB) *(by glob pattern)*
- `Data/08_remote_sensing/08_05_HyperSpectral/Tanger/natural-lands/20250221_102048_16_4001/20250221_102048_16_4001_ortho_sr_hdf5.h5` (904.3 MB) *(by glob pattern)*

**09_05_05_biophysical_and_biochemical_trait_retrieval.ipynb**

- `Data/08_remote_sensing/08_05_HyperSpectral/Tanger/agriculture/20250224_145149_32_4001/20250224_145149_32_4001_ortho_radiance_hdf5.h5` (459.6 MB) *(by glob pattern)*
- `Data/08_remote_sensing/08_05_HyperSpectral/Tanger/agriculture/20250224_145149_32_4001/20250224_145149_32_4001_ortho_sr_hdf5.h5` (857.8 MB) *(by glob pattern)*

**09_05_06_spectral_libraries_and_spectral_matching.ipynb**

- `Data/08_remote_sensing/08_05_HyperSpectral/Tanger/energy-mining/20250205_175845_00_4001/20250205_175845_00_4001_ortho_radiance_hdf5.h5` (661.6 MB) *(by glob pattern)*
- `Data/08_remote_sensing/08_05_HyperSpectral/Tanger/energy-mining/20250205_175845_00_4001/20250205_175845_00_4001_ortho_sr_hdf5.h5` (1.2 GB) *(by glob pattern)*
- `Data/08_remote_sensing/08_05_HyperSpectral/Tanger/urban/20250423_193451_57_4001/20250423_193451_57_4001_ortho_radiance_hdf5.h5` (629.5 MB) *(by glob pattern)*
- `Data/08_remote_sensing/08_05_HyperSpectral/Tanger/urban/20250423_193451_57_4001/20250423_193451_57_4001_ortho_sr_hdf5.h5` (1.2 GB) *(by glob pattern)*

**09_05_07_endmember_extraction_and_spectral_unmixing.ipynb**

- `Data/08_remote_sensing/08_05_HyperSpectral/Pavia/Pavia_gt.mat` (34.1 KB)
- `Data/08_remote_sensing/08_05_HyperSpectral/Pavia/pavia.npy` (152.5 MB)
- `Data/08_remote_sensing/08_05_HyperSpectral/Tanger/urban/20250423_193451_57_4001/20250423_193451_57_4001_ortho_radiance_hdf5.h5` (629.5 MB) *(by glob pattern)*
- `Data/08_remote_sensing/08_05_HyperSpectral/Tanger/urban/20250423_193451_57_4001/20250423_193451_57_4001_ortho_sr_hdf5.h5` (1.2 GB) *(by glob pattern)*

**09_05_08_target_and_anomaly_detection.ipynb**

- `Data/08_remote_sensing/08_05_HyperSpectral/Tanger/energy-mining/20250205_175845_00_4001/20250205_175845_00_4001_ortho_radiance_hdf5.h5` (661.6 MB) *(by glob pattern)*
- `Data/08_remote_sensing/08_05_HyperSpectral/Tanger/energy-mining/20250205_175845_00_4001/20250205_175845_00_4001_ortho_sr_hdf5.h5` (1.2 GB) *(by glob pattern)*
- `Data/08_remote_sensing/08_05_HyperSpectral/Tanger/urban/20250423_193451_57_4001/20250423_193451_57_4001_ortho_radiance_hdf5.h5` (629.5 MB) *(by glob pattern)*
- `Data/08_remote_sensing/08_05_HyperSpectral/Tanger/urban/20250423_193451_57_4001/20250423_193451_57_4001_ortho_sr_hdf5.h5` (1.2 GB) *(by glob pattern)*

**09_05_09_greenhouse_gas_plume_detection.ipynb**

- `Data/08_remote_sensing/08_05_HyperSpectral/Tanger/GHG-plumes/20241004_081921_28_4001/20241004_081921_28_4001_ortho_ql_ch4.tif` (35.8 KB) *(by glob pattern)*
- `Data/08_remote_sensing/08_05_HyperSpectral/Tanger/GHG-plumes/20241004_081921_28_4001/20241004_081921_28_4001_ortho_radiance_hdf5.h5` (682.3 MB) *(by glob pattern)*
- `Data/08_remote_sensing/08_05_HyperSpectral/Tanger/GHG-plumes/20241004_081921_28_4001/20241004_081921_28_4001_ortho_sr_hdf5.h5` (1.3 GB) *(by glob pattern)*
- `Data/08_remote_sensing/08_05_HyperSpectral/Tanger/GHG-plumes/20241004_081921_28_4001/20241004_081921_28_4001_ql_ch4_json.geojson` (8.3 KB) *(by glob pattern)*

**09_05_10_thematic_applications_across_tanager_scenes.ipynb**

- `Data/08_remote_sensing/08_05_HyperSpectral/Tanger/coastal-water-bodies/20250225_153627_16_4001/20250225_153627_16_4001_ortho_radiance_hdf5.h5` (459.9 MB) *(by glob pattern)*
- `Data/08_remote_sensing/08_05_HyperSpectral/Tanger/coastal-water-bodies/20250225_153627_16_4001/20250225_153627_16_4001_ortho_sr_hdf5.h5` (838.0 MB) *(by glob pattern)*
- `Data/08_remote_sensing/08_05_HyperSpectral/Tanger/fire/20241215_185916_33_4001/20241215_185916_33_4001_ortho_radiance_hdf5.h5` (429.2 MB) *(by glob pattern)*
- `Data/08_remote_sensing/08_05_HyperSpectral/Tanger/fire/20241215_185916_33_4001/20241215_185916_33_4001_ortho_sr_hdf5.h5` (784.8 MB) *(by glob pattern)*
- `Data/08_remote_sensing/08_05_HyperSpectral/Tanger/natural-lands/20250221_102048_16_4001/20250221_102048_16_4001_ortho_radiance_hdf5.h5` (483.0 MB) *(by glob pattern)*
- `Data/08_remote_sensing/08_05_HyperSpectral/Tanger/natural-lands/20250221_102048_16_4001/20250221_102048_16_4001_ortho_sr_hdf5.h5` (904.3 MB) *(by glob pattern)*
- `Data/08_remote_sensing/08_05_HyperSpectral/Tanger/snow-ice/20250411_163421_16_4001/20250411_163421_16_4001_ortho_radiance_hdf5.h5` (653.8 MB) *(by glob pattern)*
- `Data/08_remote_sensing/08_05_HyperSpectral/Tanger/snow-ice/20250411_163421_16_4001/20250411_163421_16_4001_ortho_sr_hdf5.h5` (1.2 GB) *(by glob pattern)*

**09_05_11_deep_learning_for_hyperspectral_classification.ipynb**

- `Data/08_remote_sensing/08_05_HyperSpectral/Pavia/Pavia_gt.mat` (34.1 KB)
- `Data/08_remote_sensing/08_05_HyperSpectral/Pavia/pavia.npy` (152.5 MB)
- `Data/08_remote_sensing/08_05_HyperSpectral/Tanger/GHG-plumes/20241004_081921_28_4001/20241004_081921_28_4001_ortho_radiance_hdf5.h5` (682.3 MB) *(by glob pattern)*
- `Data/08_remote_sensing/08_05_HyperSpectral/Tanger/GHG-plumes/20241004_081921_28_4001/20241004_081921_28_4001_ortho_sr_hdf5.h5` (1.3 GB) *(by glob pattern)*
- `Data/08_remote_sensing/08_05_HyperSpectral/Tanger/ROCX2025/20250919_170233_04_4001/20250919_170233_04_4001_ortho_radiance_hdf5.h5` (682.5 MB) *(by glob pattern)*
- `Data/08_remote_sensing/08_05_HyperSpectral/Tanger/ROCX2025/20250919_170233_04_4001/20250919_170233_04_4001_ortho_sr_hdf5.h5` (1.2 GB) *(by glob pattern)*
- `Data/08_remote_sensing/08_05_HyperSpectral/Tanger/agriculture/20250224_145149_32_4001/20250224_145149_32_4001_ortho_radiance_hdf5.h5` (459.6 MB) *(by glob pattern)*
- `Data/08_remote_sensing/08_05_HyperSpectral/Tanger/agriculture/20250224_145149_32_4001/20250224_145149_32_4001_ortho_sr_hdf5.h5` (857.8 MB) *(by glob pattern)*
- `Data/08_remote_sensing/08_05_HyperSpectral/Tanger/coastal-water-bodies/20250225_153627_16_4001/20250225_153627_16_4001_ortho_radiance_hdf5.h5` (459.9 MB) *(by glob pattern)*
- `Data/08_remote_sensing/08_05_HyperSpectral/Tanger/coastal-water-bodies/20250225_153627_16_4001/20250225_153627_16_4001_ortho_sr_hdf5.h5` (838.0 MB) *(by glob pattern)*
- `Data/08_remote_sensing/08_05_HyperSpectral/Tanger/energy-mining/20250205_175845_00_4001/20250205_175845_00_4001_ortho_radiance_hdf5.h5` (661.6 MB) *(by glob pattern)*
- `Data/08_remote_sensing/08_05_HyperSpectral/Tanger/energy-mining/20250205_175845_00_4001/20250205_175845_00_4001_ortho_sr_hdf5.h5` (1.2 GB) *(by glob pattern)*
- `Data/08_remote_sensing/08_05_HyperSpectral/Tanger/fire/20241215_185916_33_4001/20241215_185916_33_4001_ortho_radiance_hdf5.h5` (429.2 MB) *(by glob pattern)*
- `Data/08_remote_sensing/08_05_HyperSpectral/Tanger/fire/20241215_185916_33_4001/20241215_185916_33_4001_ortho_sr_hdf5.h5` (784.8 MB) *(by glob pattern)*
- `Data/08_remote_sensing/08_05_HyperSpectral/Tanger/natural-lands/20250221_102048_16_4001/20250221_102048_16_4001_ortho_radiance_hdf5.h5` (483.0 MB) *(by glob pattern)*
- `Data/08_remote_sensing/08_05_HyperSpectral/Tanger/natural-lands/20250221_102048_16_4001/20250221_102048_16_4001_ortho_sr_hdf5.h5` (904.3 MB) *(by glob pattern)*
- `Data/08_remote_sensing/08_05_HyperSpectral/Tanger/snow-ice/20250411_163421_16_4001/20250411_163421_16_4001_ortho_radiance_hdf5.h5` (653.8 MB) *(by glob pattern)*
- `Data/08_remote_sensing/08_05_HyperSpectral/Tanger/snow-ice/20250411_163421_16_4001/20250411_163421_16_4001_ortho_sr_hdf5.h5` (1.2 GB) *(by glob pattern)*
- `Data/08_remote_sensing/08_05_HyperSpectral/Tanger/urban/20250423_193451_57_4001/20250423_193451_57_4001_ortho_radiance_hdf5.h5` (629.5 MB) *(by glob pattern)*
- `Data/08_remote_sensing/08_05_HyperSpectral/Tanger/urban/20250423_193451_57_4001/20250423_193451_57_4001_ortho_sr_hdf5.h5` (1.2 GB) *(by glob pattern)*

### 09_06_lidar

**08_06_01_lidar_fundamentals.ipynb**

- `Data/08_remote_sensing/08_06_lidar/autzen_window.laz` (7.4 MB) *(by glob pattern)*
- `Data/08_remote_sensing/08_06_lidar/madison.copc.laz` (22.1 MB) *(by file name)*
- `Data/08_remote_sensing/08_06_lidar/madison_tile_utm.laz` (787.8 KB) *(by glob pattern)*
- `Data/08_remote_sensing/08_06_lidar/madison_utm.laz` (23.8 MB) *(by glob pattern)*
- `Data/08_remote_sensing/08_06_lidar/Ithaca/`: 10 files (.las), 4.6 GB
- `Data/08_remote_sensing/08_06_lidar/UB/20080417_10851095_ld_p28.laz` (8.3 MB) *(by glob pattern)*
- `Data/08_remote_sensing/08_06_lidar/UB/20080417_10901090_ld_p28.las` (69.4 MB) *(by file name)*
- `Data/08_remote_sensing/08_06_lidar/UB/20080417_10901090_ld_p28.laz` (7.8 MB) *(by glob pattern)*
- `Data/08_remote_sensing/08_06_lidar/UB/20080417_10901095_ld_p28.laz` (8.7 MB) *(by glob pattern)*
- `Data/08_remote_sensing/08_06_lidar/UB/20080417_10951090_ld_p28.laz` (7.2 MB) *(by glob pattern)*
- `Data/08_remote_sensing/08_06_lidar/UB/20080417_10951095_ld_p28.laz` (9.1 MB) *(by glob pattern)*
- download: <https://raw.githubusercontent.com>

**09_06_02_lidar_processing_quality_control_cleaning_ground_classification_and_surface_models.ipynb**

- `Data/08_remote_sensing/08_06_lidar/madison.copc.laz` (22.1 MB) *(by file name)*
- `Data/08_remote_sensing/08_06_lidar/madison_utm.laz` (23.8 MB) *(by file name)*
- `Data/08_remote_sensing/08_06_lidar/UB/20080417_10901090_ld_p28.laz` (7.8 MB) *(by file name)*
- `Data/08_remote_sensing/08_06_lidar/UB/20080417_10951090_ld_p28.laz` (7.2 MB) *(by file name)*

**09_06_04_forest_structure_and_individual_trees_from_lidar.ipynb**

- `Data/08_remote_sensing/08_06_lidar/madison.copc.laz` (22.1 MB) *(by file name)*
- `Data/08_remote_sensing/08_06_lidar/madison_utm.laz` (23.8 MB) *(by file name)*
- `Data/08_remote_sensing/08_06_lidar/Ithaca/u_3780069650.las` (492.1 MB) *(by file name)*

**09_06_05_buildings_and_urban_features_from_lidar.ipynb**

- `Data/08_remote_sensing/08_06_lidar/autzen_window.laz` (7.4 MB) *(by file name)*
- `Data/08_remote_sensing/08_06_lidar/UB/20080417_10901090_ld_p28.laz` (7.8 MB) *(by file name)*
- `Data/08_remote_sensing/08_06_lidar/UB/20080417_10951090_ld_p28.laz` (7.2 MB) *(by file name)*

**09_06_06_point_cloud_classification_with_machine_learning_and_deep_learning.ipynb**

- `Data/08_remote_sensing/08_06_lidar/autzen_window.laz` (7.4 MB) *(by file name)*

**09_06_07_spaceborne_lidar_gedi_and_icesat_2.ipynb**

- `Data/08_remote_sensing/08_06_lidar/ATL08_simulated_madison.h5` (10.0 KB) *(by file name)*
- `Data/08_remote_sensing/08_06_lidar/GEDI02_A_simulated_madison.h5` (89.1 KB) *(by file name)*
- `Data/08_remote_sensing/08_06_lidar/madison.copc.laz` (22.1 MB) *(by file name)*
- `Data/08_remote_sensing/08_06_lidar/madison_utm.laz` (23.8 MB) *(by file name)*
- download: <https://cmr.earthdata.nasa.gov>
- download: <https://cmr.earthdata.nasa.gov/search/granules.json>

**09_06_08_large_area_lidar_processing_tiling_streaming_parallel_work_and_the_gpu.ipynb**

- `Data/08_remote_sensing/08_06_lidar/Ithaca/`: 10 files (.las), 4.6 GB

### 09_07_texture_analysis

**09_07_01_texture_analysis_of_landsat_imagery_statistical_and_structural_methods.ipynb**

- `Data/00_shared/data_remote_sensing/landsat9_vegas_20240708/LC09_L2SP_039035_20240708_02_T1/*.tif` (0 B)

**09_07_02_texture_analysis_of_landsat_imagery_transform_based_spectral_methods.ipynb**

- `Data/00_shared/data_remote_sensing/landsat9_vegas_20240708/LC09_L2SP_039035_20240708_02_T1/*.tif` (0 B)

**09_07_03_texture_analysis_of_landsat_imagery_local_pattern_based_methods_lbp.ipynb**

- `Data/00_shared/data_remote_sensing/landsat9_vegas_20240708/LC09_L2SP_039035_20240708_02_T1/*.tif` (0 B)

**09_07_04_texture_analysis_of_landsat_imagery_model_based_methods.ipynb**

- `Data/00_shared/data_remote_sensing/landsat9_vegas_20240708/LC09_L2SP_039035_20240708_02_T1/*.tif` (0 B)

**09_07_05_texture_analysis_of_landsat_imagery_deep_learning_based_methods.ipynb**

- `Data/00_shared/data_remote_sensing/landsat9_vegas_20240708/LC09_L2SP_039035_20240708_02_T1/*.tif` (0 B)

### 09_08_image_super_resolution

**09_08_01_landsat_8_to_sentinel_2_six_band_super_resolution.ipynb**

- `Data/08_remote_sensing/08_08_image_super_resolution/Landsat8_20190709/clear_mask.tif` (1.5 KB) *(by file name)*
- `Data/08_remote_sensing/08_08_image_super_resolution/Landsat8_20190709/landsat8_20190709_surface_reflectance_30m.tif` (699.7 KB) *(by file name)*
- `Data/08_remote_sensing/08_08_image_super_resolution/Landsat8_20190709/metadata.json` (1.0 KB) *(by file name)*
- `Data/08_remote_sensing/08_08_image_super_resolution/Sentinel_2/L2A_T17TQH_20190708/clear_mask_10m.tif` (3.6 KB) *(by file name)*
- `Data/08_remote_sensing/08_08_image_super_resolution/Sentinel_2/L2A_T17TQH_20190708/sentinel2_20190708_surface_reflectance_6band_10m.tif` (5.0 MB) *(by file name)*

**09_08_02_sentinel_2_to_planetscope_multispectral_super_resolution.ipynb**

- `Data/08_remote_sensing/08_08_image_super_resolution/Landsat8_20190709/metadata.json` (1.0 KB) *(by file name)*
- `Data/08_remote_sensing/08_08_image_super_resolution/Planet/July_08_2019/Planet_Rural_AOI_July_08_2019.tif` (22.6 MB) *(by file name)*
- `Data/08_remote_sensing/08_08_image_super_resolution/Sentinel_2/L2A_T17TQH_20190708/clear_mask_10m.tif` (3.6 KB) *(by file name)*
- `Data/08_remote_sensing/08_08_image_super_resolution/Sentinel_2/L2A_T17TQH_20190708/sentinel2_20190708_surface_reflectance_6band_10m.tif` (5.0 MB) *(by file name)*

**09_08_03_planetscope_to_high_resolution_naip_style_imagery_super_resolution.ipynb**

- `Data/08_remote_sensing/08_08_image_super_resolution/HR/HR_Rural_AOI_July_14.tif` (343.1 MB) *(by file name)*
- `Data/08_remote_sensing/08_08_image_super_resolution/Planet/July_08_2019/Rural_July_08_2019_PSScene4Band_Explorer/manifest.json` (2.1 KB) *(by file name)*
- `Data/08_remote_sensing/08_08_image_super_resolution/Planet/July_14_2019/Planet_Rural_AOI_July_14_2019_NAD.tif` (26.4 MB) *(by file name)*
- `Data/08_remote_sensing/08_08_image_super_resolution/Planet/July_14_2019/Rural_July_14_2019_PSScene4Band_Explorer/files/PSScene4Band/20190714_142708_1049/20190714_142708_1049_metadata.json` (1.1 KB) *(by file name)*

**09_08_04_landsat_8_sentinel_2_planetscope_and_high_resolution_imagery.ipynb**

- `Data/08_remote_sensing/08_08_image_super_resolution/HR/HR_Rural_AOI_July_14.tif` (343.1 MB) *(by file name)*
- `Data/08_remote_sensing/08_08_image_super_resolution/Landsat8_20190709/clear_mask.tif` (1.5 KB) *(by file name)*
- `Data/08_remote_sensing/08_08_image_super_resolution/Landsat8_20190709/landsat8_20190709_surface_reflectance_30m.tif` (699.7 KB) *(by file name)*
- `Data/08_remote_sensing/08_08_image_super_resolution/Landsat8_20190709/metadata.json` (1.0 KB) *(by file name)*
- `Data/08_remote_sensing/08_08_image_super_resolution/Planet/July_08_2019/Rural_July_08_2019_PSScene4Band_Explorer/manifest.json` (2.1 KB) *(by file name)*
- `Data/08_remote_sensing/08_08_image_super_resolution/Planet/July_14_2019/Planet_Rural_AOI_July_14_2019_NAD.tif` (26.4 MB) *(by file name)*
- `Data/08_remote_sensing/08_08_image_super_resolution/Planet/July_14_2019/Rural_July_14_2019_PSScene4Band_Explorer/files/PSScene4Band/20190714_142708_1049/20190714_142708_1049_metadata.json` (1.1 KB) *(by file name)*
- `Data/08_remote_sensing/08_08_image_super_resolution/Sentinel_2/L2A_T17TQH_20190708/clear_mask_10m.tif` (3.6 KB) *(by file name)*
- `Data/08_remote_sensing/08_08_image_super_resolution/Sentinel_2/L2A_T17TQH_20190708/sentinel2_20190708_surface_reflectance_6band_10m.tif` (5.0 MB) *(by file name)*

### 09_09_image_classification

**09_09_01_01_introduction_to_image_classification.ipynb**

- `Data/08_remote_sensing/08_09_image_classification/Sentinel_2_UB/Label_Point_Data/point_data.csv` (2.5 MB)

**09_09_01_02_image_processing.ipynb**

- `Data/08_remote_sensing/08_09_image_classification/Edge_Detection/BD_WorldView/multi.tif` (8.7 MB) *(by file name)*
- `Data/08_remote_sensing/08_09_image_classification/Image_processing/DG_MAR.tif` (114.2 MB)
- `Data/08_remote_sensing/08_09_image_classification/Image_processing/crop_field.jpg` (77.7 KB)
- `Data/08_remote_sensing/08_09_image_classification/Image_processing/sw_030.TIF` (95.6 MB)
- `Data/08_remote_sensing/08_09_image_classification/Image_processing/Planet/before/B*.tif` (0 B)

**09_09_01_03_sampling_accuracy_assessment_and_reproducible_runs.ipynb**

- `Data/00_shared/data_remote_sensing/Data_Ithaca/NAIP/naip_2026-08-17_2026-10-16/stack.tif` (11.5 MB) *(by file name)*
- `Data/08_remote_sensing/08_09_image_classification/Sentinel_01_UB/Bands/VV.tif` (559.7 KB)
- `Data/08_remote_sensing/08_09_image_classification/Sentinel_2_UB/Images/B2.tif` (82.2 KB)
- `Data/08_remote_sensing/08_09_image_classification/Sentinel_2_UB/Label_Point_Data/GroundTruth_data.shp` (15.2 KB)
- `Data/08_remote_sensing/08_09_image_classification/Sentinel_2_UB/Label_Point_Data/point_data.csv` (2.5 MB)

**09_09_02_01_classical_image_classification_pixel_based_and_object_based.ipynb**

- `Data/08_remote_sensing/08_09_image_classification/Corn_belt/CDL/10_m/cdl_class_legend.csv` (3.7 KB) *(by file name)*
- `Data/08_remote_sensing/08_09_image_classification/Corn_belt/CDL/30_m/CDL_2025_30m.tif` (83.0 KB) *(by file name)*
- `Data/08_remote_sensing/08_09_image_classification/Corn_belt/Landsat_9/2020_09_LC08_L2SP_027030_20200903_02_T1/aoi.geojson` (440 B) *(by file name)*
- `Data/08_remote_sensing/08_09_image_classification/Corn_belt/Landsat_9/2020_09_LC08_L2SP_027030_20200903_02_T1/request.json` (612 B) *(by file name)*

**09_09_02_02_image_classification_with_a_cnn_on_the_gpu_landsat_and_the_cropland_data_layer.ipynb**

- `Data/08_remote_sensing/08_09_image_classification/Corn_belt/CDL/30_m/CDL_2025_30m.tif` (83.0 KB) *(by file name)*
- `Data/08_remote_sensing/08_09_image_classification/Corn_belt/Landsat_9/2020_09_LC08_L2SP_027030_20200903_02_T1/aoi.geojson` (440 B) *(by file name)*
- `Data/08_remote_sensing/08_09_image_classification/Corn_belt/Landsat_9/2020_09_LC08_L2SP_027030_20200903_02_T1/qa_pixel.tif` (6.6 KB) *(by file name)*
- `Data/08_remote_sensing/08_09_image_classification/Corn_belt/Landsat_9/2020_09_LC08_L2SP_027030_20200903_02_T1/request.json` (612 B) *(by file name)*
- `Data/08_remote_sensing/08_09_image_classification/Corn_belt/outside_aoi/aoi_outside.geojson` (245 B) *(by file name)*

**09_09_02_03_deep_learning_for_sentinel_2_cnn_resnet_efficientnet_convnext_vit_and_swin_on_the_gpu.ipynb**

- `Data/08_remote_sensing/08_09_image_classification/Corn_belt/CDL/10_m/CDL_2025_10m.tif` (331.4 KB) *(by file name)*
- `Data/08_remote_sensing/08_09_image_classification/Corn_belt/Landsat_9/2020_09_LC08_L2SP_027030_20200903_02_T1/aoi.geojson` (440 B) *(by file name)*
- `Data/08_remote_sensing/08_09_image_classification/Corn_belt/Landsat_9/2020_09_LC08_L2SP_027030_20200903_02_T1/red.tif` (1.5 MB) *(by file name)*
- `Data/08_remote_sensing/08_09_image_classification/Corn_belt/Sentinel_2/2020_06_20200625/scl.tif` (97.3 KB) *(by file name)*

**09_09_02_04_transfer_learning_and_vision_transformers_for_sentinel_2_imagenet_vs_remote_sensing_weights.ipynb**

- `Data/08_remote_sensing/08_09_image_classification/Corn_belt/CDL/10_m/CDL_2025_10m.tif` (331.4 KB) *(by file name)*
- `Data/08_remote_sensing/08_09_image_classification/Corn_belt/Landsat_9/2020_09_LC08_L2SP_027030_20200903_02_T1/aoi.geojson` (440 B) *(by file name)*
- `Data/08_remote_sensing/08_09_image_classification/Corn_belt/Landsat_9/2020_09_LC08_L2SP_027030_20200903_02_T1/red.tif` (1.5 MB) *(by file name)*
- `Data/08_remote_sensing/08_09_image_classification/Corn_belt/Sentinel_2/2020_06_20200625/scl.tif` (97.3 KB) *(by file name)*

**09_09_03_01_crop_classification_with_sentinel_1_sar_time_series_pixel_classifiers_and_deep_cnns_vv_vh_may_october_2024.ipynb**

- `Data/08_remote_sensing/08_09_image_classification/Corn_belt/CDL/10_m/CDL_2024_10m.tif` (313.1 KB) *(by file name)*
- `Data/08_remote_sensing/08_09_image_classification/Corn_belt/CDL/10_m/inventory.csv` (360 B) *(by file name)*
- `Data/08_remote_sensing/08_09_image_classification/Corn_belt/Landsat_9/2020_09_LC08_L2SP_027030_20200903_02_T1/aoi.geojson` (440 B) *(by file name)*

**09_09_03_02_spatio_temporal_and_multi_sensor_crop_mapping_sentinel_2_sentinel_1_and_landsat_time_series_2020_to_2025.ipynb**

- `Data/08_remote_sensing/08_09_image_classification/Corn_belt/2025_cultivated_layer/2025_cultivated_layer.tif` (341.6 MB) *(by file name)*
- `Data/08_remote_sensing/08_09_image_classification/Corn_belt/Landsat_9/2020_09_LC08_L2SP_027030_20200903_02_T1/red.tif` (1.5 MB) *(by file name)*
- `Data/08_remote_sensing/08_09_image_classification/Corn_belt/Landsat_9/2020_09_LC08_L2SP_027030_20200903_02_T1/swir16.tif` (1.6 MB) *(by file name)*
- `Data/08_remote_sensing/08_09_image_classification/Corn_belt/Sentinel_2/2020_06_20200625/scl.tif` (97.3 KB) *(by file name)*
- `Data/08_remote_sensing/08_09_image_classification/Corn_belt/cache_20m/labels_20m.nc` (27.4 MB) *(by file name)*
- `Data/08_remote_sensing/08_09_image_classification/Corn_belt/cache_20m/s1_20m.nc` (409.7 MB) *(by file name)*
- `Data/08_remote_sensing/08_09_image_classification/Corn_belt/cache_20m/s2_20m.nc` (2.0 GB) *(by file name)*
- `Data/08_remote_sensing/08_09_image_classification/Corn_belt/outputs/spatio_temporal/cloud_experiment.csv` (2.8 KB) *(by file name)* *(written by a notebook)*
- `Data/08_remote_sensing/08_09_image_classification/Corn_belt/outputs/spatio_temporal/crop_map_2025_early_fusion_cloudaug_20m.tif` (175.0 KB) *(by file name)* *(written by a notebook)*
- `Data/08_remote_sensing/08_09_image_classification/Corn_belt/outputs/spatio_temporal/in_season.csv` (2.3 KB) *(by file name)* *(written by a notebook)*
- `Data/08_remote_sensing/08_09_image_classification/Corn_belt/outputs/spatio_temporal/test_scores_pixel_models.csv` (807 B) *(by file name)* *(written by a notebook)*

**09_09_03_03_change_detection_classical_post_classification_and_deep_methods_on_landsat_and_sentinel_2_2020_to_2025.ipynb**

- `Data/08_remote_sensing/08_09_image_classification/Corn_belt/Landsat_9/2020_09_LC08_L2SP_027030_20200903_02_T1/qa_pixel.tif` (6.6 KB) *(by file name)*
- `Data/08_remote_sensing/08_09_image_classification/Corn_belt/Landsat_9/2020_09_LC08_L2SP_027030_20200903_02_T1/red.tif` (1.5 MB) *(by file name)*
- `Data/08_remote_sensing/08_09_image_classification/Corn_belt/Sentinel_2/2020_06_20200625/scl.tif` (97.3 KB) *(by file name)*
- `Data/08_remote_sensing/08_09_image_classification/Corn_belt/outputs/change_detection/change_metrics_2024_2025.csv` (4.7 KB) *(by file name)* *(written by a notebook)*
- `Data/08_remote_sensing/08_09_image_classification/Corn_belt/outputs/change_detection/change_probability_s2_2024_2025.tif` (395.0 KB) *(by file name)* *(written by a notebook)*
- `Data/08_remote_sensing/08_09_image_classification/Corn_belt/outputs/change_detection/olofsson_change_area.csv` (537 B) *(by file name)* *(written by a notebook)*
- `Data/08_remote_sensing/08_09_image_classification/Corn_belt/outputs/change_detection/transitions_landsat_2024_2025.tif` (347.0 KB) *(by file name)* *(written by a notebook)*

**09_09_04_01_semantic_segmentation_of_crop_types_from_sentinel_2_fcn_u_net_deeplabv3_plus_and_segformer_on_the_gpu.ipynb**

- `Data/08_remote_sensing/08_09_image_classification/Corn_belt/CDL/10_m/CDL_2025_10m.tif` (331.4 KB) *(by file name)*
- `Data/08_remote_sensing/08_09_image_classification/Corn_belt/Landsat_9/2020_09_LC08_L2SP_027030_20200903_02_T1/aoi.geojson` (440 B) *(by file name)*
- `Data/08_remote_sensing/08_09_image_classification/Corn_belt/Landsat_9/2020_09_LC08_L2SP_027030_20200903_02_T1/red.tif` (1.5 MB) *(by file name)*
- `Data/08_remote_sensing/08_09_image_classification/Corn_belt/Sentinel_2/2020_06_20200625/scl.tif` (97.3 KB) *(by file name)*
- `Data/08_remote_sensing/08_09_image_classification/Corn_belt/outputs/s2_20250826_segmentation.tif` (345.9 KB) *(by file name)* *(written by a notebook)*
- `Data/08_remote_sensing/08_09_image_classification/Corn_belt/outputs/s2_20250826_semantic_segmentation_outside_aoi.tif` (87.4 KB) *(by file name)* *(written by a notebook)*

**09_09_04_02_instance_segmentation_of_agricultural_fields_with_mask_r_cnn.ipynb**

- `Data/08_remote_sensing/08_09_image_classification/WorldView2_BD/Labeled_data/aoi_trianing.shp` (105.7 KB) *(by file name)*
- `Data/08_remote_sensing/08_09_image_classification/WorldView2_BD/Labeled_data/multi.tif` (8.7 MB) *(by file name)*
- `Data/08_remote_sensing/08_09_image_classification/WorldView2_BD/outputs/instance_segmentation/mask_rcnn_test_instances.gpkg` (96.0 KB) *(by file name)* *(written by a notebook)*

**09_09_05_03_imperfect_labels_and_domain_shift.ipynb**

- `Data/08_remote_sensing/08_09_image_classification/Corn_belt/cache_20m/labels_20m.nc` (27.4 MB) *(by file name)*
- `Data/08_remote_sensing/08_09_image_classification/Corn_belt/cache_20m/s2_20m.nc` (2.0 GB) *(by file name)*
- `Data/08_remote_sensing/08_09_image_classification/Image_processing/Planet/before/B2.tif` (78.8 MB) *(by file name)*
- `Data/08_remote_sensing/08_09_image_classification/Sentinel_2_UB/Label_Point_Data/point_data.csv` (2.5 MB) *(by file name)*

**09_09_05_04_multi_sensor_image_fusion.ipynb**

- `Data/08_remote_sensing/08_09_image_classification/Corn_belt/Sentinel_1/2020_05_20200517_63/vv.tif` (29.8 MB) *(by file name)*
- `Data/08_remote_sensing/08_09_image_classification/Image_Fusion/AOI/rural_aoi_GCS.shp` (236 B) *(by file name)*
- `Data/08_remote_sensing/08_09_image_classification/Image_Fusion/CDL_2019/cropland_2019_1423905_2362575_90000` (61.0 KB) *(by glob pattern)*
- `Data/08_remote_sensing/08_09_image_classification/Image_Fusion/CDL_2019/cropland_2019_1423905_2362575_90000/confidence.tif` (32.8 KB) *(by file name)*
- `Data/08_remote_sensing/08_09_image_classification/Image_Fusion/CDL_2019/cropland_2019_1423905_2362575_90000/cropland.tif` (11.9 KB) *(by file name)*
- `Data/08_remote_sensing/08_09_image_classification/Image_Fusion/CDL_2019/cropland_2019_1423905_2362575_90000/item.json` (16.3 KB) *(by glob pattern)*
- `Data/08_remote_sensing/08_09_image_classification/Image_Fusion/HR/HR_Rural_AOI_July_14.tif` (343.1 MB) *(by file name)*
- `Data/08_remote_sensing/08_09_image_classification/Image_Fusion/Landsat8_20190709/clear_mask.tif` (1.5 KB) *(by file name)*
- `Data/08_remote_sensing/08_09_image_classification/Image_Fusion/Landsat8_20190709/landsat8_20190709_surface_reflectance_30m.tif` (699.7 KB) *(by file name)*
- `Data/08_remote_sensing/08_09_image_classification/Image_Fusion/MODIS_MOD09A1_2019/`: 7 files (.2020298231428, .2020302064635, .2020303101652, .2020303202254, .2020304045534, .2020304103711, .2020304200305), 823.2 KB
- `Data/08_remote_sensing/08_09_image_classification/Image_Fusion/MODIS_MOD09A1_2019/MOD09A1.A2019169.h12v04.061.2020298231428/item.json` (16.7 KB) *(by glob pattern)*
- `Data/08_remote_sensing/08_09_image_classification/Image_Fusion/MODIS_MOD09A1_2019/MOD09A1.A2019169.h12v04.061.2020298231428/sur_refl_state_500m.tif` (14.9 KB) *(by file name)*
- `Data/08_remote_sensing/08_09_image_classification/Image_Fusion/MODIS_MOD09A1_2019/MOD09A1.A2019177.h12v04.061.2020302064635/item.json` (16.7 KB) *(by glob pattern)*
- `Data/08_remote_sensing/08_09_image_classification/Image_Fusion/MODIS_MOD09A1_2019/MOD09A1.A2019185.h12v04.061.2020303101652/item.json` (16.7 KB) *(by glob pattern)*
- `Data/08_remote_sensing/08_09_image_classification/Image_Fusion/MODIS_MOD09A1_2019/MOD09A1.A2019193.h12v04.061.2020303202254/item.json` (16.7 KB) *(by glob pattern)*
- `Data/08_remote_sensing/08_09_image_classification/Image_Fusion/MODIS_MOD09A1_2019/MOD09A1.A2019201.h12v04.061.2020304045534/item.json` (16.7 KB) *(by glob pattern)*
- `Data/08_remote_sensing/08_09_image_classification/Image_Fusion/MODIS_MOD09A1_2019/MOD09A1.A2019209.h12v04.061.2020304103711/item.json` (16.7 KB) *(by glob pattern)*
- `Data/08_remote_sensing/08_09_image_classification/Image_Fusion/MODIS_MOD09A1_2019/MOD09A1.A2019217.h12v04.061.2020304200305/item.json` (16.7 KB) *(by glob pattern)*
- `Data/08_remote_sensing/08_09_image_classification/Image_Fusion/Planet/July_08_2019/Planet_Rural_AOI_July_08_2019.tif` (22.6 MB) *(by file name)*
- `Data/08_remote_sensing/08_09_image_classification/Image_Fusion/Planet/July_14_2019/Planet_Rural_AOI_July_14_2019_NAD.tif` (26.4 MB) *(by file name)*
- `Data/08_remote_sensing/08_09_image_classification/Image_Fusion/RapidEye/RapidEye_Rural_AOI_July_24_2019.tif` (11.1 MB) *(by file name)*
- `Data/08_remote_sensing/08_09_image_classification/Image_Fusion/Sentinel_1_RTC_2019/S1A_IW_GRDH_1SDV_20190629T230759_20190629T230824_027901_032662_rtc` (2.4 MB) *(by glob pattern)*
- `Data/08_remote_sensing/08_09_image_classification/Image_Fusion/Sentinel_1_RTC_2019/S1A_IW_GRDH_1SDV_20190711T230800_20190711T230825_028076_032BAF_rtc` (2.4 MB) *(by glob pattern)*
- `Data/08_remote_sensing/08_09_image_classification/Image_Fusion/Sentinel_1_RTC_2019/S1A_IW_GRDH_1SDV_20190723T230801_20190723T230826_028251_0330F6_rtc` (2.4 MB) *(by glob pattern)*
- `Data/08_remote_sensing/08_09_image_classification/Image_Fusion/Sentinel_1_RTC_2019/S1A_IW_GRDH_1SDV_20190804T230802_20190804T230827_028426_033652_rtc` (2.4 MB) *(by glob pattern)*
- `Data/08_remote_sensing/08_09_image_classification/Image_Fusion/Sentinel_1_RTC_2019/S1A_IW_GRDH_1SDV_20190629T230759_20190629T230824_027901_032662_rtc/item.json` (7.1 KB) *(by glob pattern)*
- `Data/08_remote_sensing/08_09_image_classification/Image_Fusion/Sentinel_1_RTC_2019/S1A_IW_GRDH_1SDV_20190711T230800_20190711T230825_028076_032BAF_rtc/item.json` (7.1 KB) *(by glob pattern)*
- `Data/08_remote_sensing/08_09_image_classification/Image_Fusion/Sentinel_1_RTC_2019/S1A_IW_GRDH_1SDV_20190723T230801_20190723T230826_028251_0330F6_rtc/item.json` (7.1 KB) *(by glob pattern)*
- `Data/08_remote_sensing/08_09_image_classification/Image_Fusion/Sentinel_1_RTC_2019/S1A_IW_GRDH_1SDV_20190804T230802_20190804T230827_028426_033652_rtc/item.json` (7.1 KB) *(by glob pattern)*
- `Data/08_remote_sensing/08_09_image_classification/Image_Fusion/Sentinel_2/L2A_T17TQH_20190708/clear_mask_10m.tif` (3.6 KB) *(by file name)*
- `Data/08_remote_sensing/08_09_image_classification/Image_Fusion/Sentinel_2/L2A_T17TQH_20190708/sentinel2_20190708_surface_reflectance_6band_10m.tif` (5.0 MB) *(by file name)*

**_wv_test.ipynb**

- `Data/08_remote_sensing/08_09_image_classification/Edge_Detection/BD_WorldView/aoi_trianing.shp` (105.7 KB) *(by file name)*
- `Data/08_remote_sensing/08_09_image_classification/Edge_Detection/BD_WorldView/multi.tif` (8.7 MB) *(by file name)*

### 09_10_object_detection

**09_10_00_object_detection.ipynb**

- `*.txt` (0 B)
- `Data/08_remote_sensing/08_10_object_detection/Edge_Detection/BD_WorldView/aoi_trianing.shp` (105.7 KB)
- `Data/08_remote_sensing/08_10_object_detection/Edge_Detection/BD_WorldView/multi.tif` (8.7 MB)
- `Data/08_remote_sensing/08_10_object_detection/Tree_Detection/test/images/*` (0 B)
- `Data/08_remote_sensing/08_10_object_detection/Tree_Detection/test/labels/*.txt` (0 B)
- `Data/08_remote_sensing/08_10_object_detection/cv2-pool-detection-master/data/*.jpg` (0 B)

**09_10_01_object_detection_in_aerial_imagery.ipynb**

- `*.txt` (0 B)
- `Data/08_remote_sensing/08_10_object_detection/cv2-pool-detection-master/data/`: 29 files (.jpg), 22.9 MB

**09_10_03_field_edge_detection_in_worldview_imagery_rgb_versus_rgb_plus_nir.ipynb**

- `Data/08_remote_sensing/08_10_object_detection/Edge_Detection/BD_WorldView/aoi_trianing.shp` (105.7 KB)
- `Data/08_remote_sensing/08_10_object_detection/Edge_Detection/BD_WorldView/multi.tif` (8.7 MB)

**09_10_04_field_edge_detection_in_sentinel_2_fields_of_the_world_versus_delineate_anything.ipynb**

- `Data/08_remote_sensing/08_09_image_classification/Corn_belt/outputs/field_edge_sentinel2/cornbelt_fields_2025_fiboa.parquet` (268.0 KB) *(by file name)* *(written by a notebook)*
- `Data/08_remote_sensing/08_10_object_detection/Corn_belt/CDL/10_m/cdl_class_legend.csv` (3.7 KB) *(by file name)*
- `Data/08_remote_sensing/08_10_object_detection/Corn_belt/CDL/10_m/inventory.csv` (360 B) *(by file name)*
- `Data/08_remote_sensing/08_10_object_detection/Corn_belt/Sentinel_2/2024_05_20240518/red.tif` (11.2 MB) *(by file name)*
- `Data/08_remote_sensing/08_10_object_detection/Corn_belt/Sentinel_2/2024_05_20240518/scl.tif` (70.6 KB) *(by file name)*
- download: <https://hf.co/torchgeo/delineate-anything-s/resolve/69cd440b0c5bd450ced145e68294aa9393ddae05/delineate_anything_s_rgb_yolo11n-b879d643.pt>
- download: <https://hf.co/torchgeo/delineate-anything/resolve/60bea7b2f81568d16d5c75e4b5b06289e1d7efaf/delineate_anything_rgb_yolo11x-88ede029.pt>
- download: <https://hf.co/MykolaL/DelineateAnything/resolve/main/DelineateAnythingv2.pt>

### 09_11_foundations_models

**09_11_00_what_foundation_models_are_and_why_remote_sensing_needs_them.ipynb**

- `Data/08_remote_sensing/08_09_image_classification/Sentinel_2_UB/Images/*.tif` (0 B)
- `Data/08_remote_sensing/08_09_image_classification/Sentinel_2_UB/Label_Point_Data/point_data.csv` (2.5 MB)

**09_11_01_self_supervised_pretraining_for_sentinel_2_crop_mapping.ipynb**

- `Data/08_remote_sensing/08_09_image_classification/Corn_belt/cache_20m/labels_20m.nc` (27.4 MB) *(by file name)*
- `Data/08_remote_sensing/08_09_image_classification/Corn_belt/cache_20m/s2_20m.nc` (2.0 GB) *(by file name)*
- `Data/08_remote_sensing/08_09_image_classification/Corn_belt/outputs/self_supervised/ssl_encoders_corn_belt.pt` (173.3 MB) *(by file name)* *(written by a notebook)*

**09_11_02_geospatial_foundation_models_embeddings_promptable_segmentation_and_zero_shot_classification.ipynb**

- `Data/08_remote_sensing/08_02_optical/bd_delta_50yr/LC08_L2SP_136045_20180129_02_T1/red.tif` (1.8 MB) *(by file name)*
- `Data/08_remote_sensing/08_09_image_classification/Corn_belt/CDL/10_m/CDL_2025_10m.tif` (331.4 KB) *(by file name)*
- `Data/08_remote_sensing/08_09_image_classification/Image_processing/Planet/before/B2.tif` (78.8 MB) *(by file name)*
- `Data/08_remote_sensing/08_09_image_classification/Landcover_data/class_dict.csv` (157 B) *(by file name)*
- `Data/08_remote_sensing/08_09_image_classification/Sentinel_2_UB/Label_Point_Data/GroundTruth_data.shp` (15.2 KB) *(by file name)*

**09_11_03_from_a_geotiff_to_an_embedding_map.ipynb**

- `Data/08_remote_sensing/08_09_image_classification/Corn_belt/cache_20m/labels_20m.nc` (27.4 MB)
- `Data/08_remote_sensing/08_09_image_classification/Corn_belt/cache_20m/s2_20m.nc` (2.0 GB)

**09_11_04_prithvi_ssl4eo_satlaspretrain_and_scale_mae_on_sentinel_2.ipynb**

- `Data/08_remote_sensing/08_09_image_classification/Corn_belt/cache_20m/labels_20m.nc` (27.4 MB)
- `Data/08_remote_sensing/08_09_image_classification/Corn_belt/cache_20m/s2_20m.nc` (2.0 GB)

**09_11_05_one_model_for_many_sensors_dofa_panopticon_copernicus_fm_croma_and_terramind.ipynb**

- `Data/08_remote_sensing/08_09_image_classification/Corn_belt/Landsat_9/2025_07_LC08_L2SP_027030_20250715_02_T1/*.tif` (0 B)
- `Data/08_remote_sensing/08_09_image_classification/Corn_belt/cache_20m/labels_20m.nc` (27.4 MB)
- `Data/08_remote_sensing/08_09_image_classification/Corn_belt/cache_20m/s1_20m.nc` (409.7 MB)
- `Data/08_remote_sensing/08_09_image_classification/Corn_belt/cache_20m/s2_20m.nc` (2.0 GB)
- `Data/08_remote_sensing/08_09_image_classification/Edge_Detection/BD_WorldView/aoi_trianing.shp` (105.7 KB) *(by file name)*
- `Data/08_remote_sensing/08_09_image_classification/WorldView2_BD/Images/multi.tif` (9.9 MB)

**09_11_06_time_in_foundation_models_prithvi_tl_anysat_presto_and_galileo.ipynb**

- `Data/08_remote_sensing/08_09_image_classification/Corn_belt/cache_20m/labels_20m.nc` (27.4 MB)
- `Data/08_remote_sensing/08_09_image_classification/Corn_belt/cache_20m/s2_20m.nc` (2.0 GB)

**09_11_07_alphaearth_foundations_precomputed_embeddings_for_the_whole_planet.ipynb**

- `Data/00_shared/data_geostatistics/Data_GP/GP_POINT_SOC.shp` (15.7 KB)
- `Data/08_remote_sensing/08_09_image_classification/Corn_belt/cache_20m/labels_20m.nc` (27.4 MB)
- `Data/08_remote_sensing/08_09_image_classification/Corn_belt/cache_20m/s2_20m.nc` (2.0 GB)

**09_11_08_foundation_model_embeddings_as_covariates_digital_soil_mapping_of_soc.ipynb**

- `Data/00_shared/data_geostatistics/Data_GP/GP_POINT_SOC.shp` (15.7 KB)
- `Data/03_spatial_analysis/Great_plain/gp_grid_new.csv` (105.5 MB)
- `Data/03_spatial_analysis/Great_plain/Raster/*.tif` (0 B)

**09_11_09_very_high_resolution_dinov3_sat_satlas_naip_scale_mae_and_sam_2.ipynb**

- `Data/08_remote_sensing/08_09_image_classification/Landcover_data/class_dict.csv` (157 B)
- `Data/08_remote_sensing/08_09_image_classification/Landcover_data/metadata.csv` (56.1 KB)
- `Data/08_remote_sensing/08_09_image_classification/WorldView2_BD/Images/multi.tif` (9.9 MB)
- `Data/08_remote_sensing/08_09_image_classification/WorldView2_BD/Labeled_data/aoi_trianing.shp` (105.7 KB)

**09_11_10_vision_language_models_zero_shot_classification_and_text_retrieval.ipynb**

- `Data/08_remote_sensing/08_09_image_classification/Landcover_data/class_dict.csv` (157 B)
- `Data/08_remote_sensing/08_09_image_classification/Landcover_data/metadata.csv` (56.1 KB)
- `Data/08_remote_sensing/08_11_Capstones/EuroSat.zip` (2.0 GB)

**09_11_11_change_and_similarity_in_embedding_space.ipynb**

- `Data/08_remote_sensing/08_02_optical/bd_delta_50yr/LC08_L2SP_136045_20180129_02_T1/item.json` (36.0 KB) *(by file name)*
- `Data/08_remote_sensing/08_02_optical/bd_delta_50yr/LC08_L2SP_136045_20180129_02_T1/red.tif` (1.8 MB) *(by file name)*
- `Data/08_remote_sensing/08_02_optical/s2_rohingya_2016_2018/S2A_MSIL2A_20160105T042142_R090_T46QCJ_20210527T085101/scl.tif` (1.1 KB) *(by file name)*
- `Data/08_remote_sensing/08_11_Capstones/bd_delta_annual/LC08_L2SP_137045_20*` (0 B)
- `Data/08_remote_sensing/08_11_Capstones/bug_fire_sentinel2/bug_fire_perimeter.geojson` (368.8 KB)
- `Data/08_remote_sensing/08_11_Capstones/bug_fire_sentinel2/post/`: 11 files (), 772.1 MB
- `Data/08_remote_sensing/08_11_Capstones/bug_fire_sentinel2/pre/`: 11 files (), 756.0 MB

**09_11_13_benchmarking_foundation_models_on_your_own_data.ipynb**

- `Data/08_remote_sensing/08_09_image_classification/Corn_belt/cache_20m/labels_20m.nc` (27.4 MB)
- `Data/08_remote_sensing/08_09_image_classification/Corn_belt/cache_20m/s2_20m.nc` (2.0 GB)
- `Data/08_remote_sensing/08_11_Capstones/EuroSat.zip` (2.0 GB)

### 09_13_Capstones

**09_13_01_thirty_five_years_of_change_around_pucallpa_peru.ipynb**

- download: <https://raw.githubusercontent.com>

**09_13_02_fifty_years_of_change_in_the_bangladesh_delta.ipynb**

- `Data/08_remote_sensing/08_02_optical/scene_list.pkl` (7.7 KB) *(by file name)*
- `Data/08_remote_sensing/08_02_optical/bd_delta_50yr/LC08_L2SP_136045_20180129_02_T1/green.tif` (1.8 MB) *(by file name)*
- `Data/08_remote_sensing/08_02_optical/bd_delta_50yr/LC08_L2SP_136045_20180129_02_T1/red.tif` (1.8 MB) *(by file name)*
- download: <https://planetarycomputer.microsoft.com/api/stac/v1>

**09_13_03_wild_fire_burned_area_mapping.ipynb**

- `Data/08_remote_sensing/08_02_optical/bd_delta_50yr/LC08_L2SP_136045_20180129_02_T1/item.json` (36.0 KB) *(by file name)*
- `Data/08_remote_sensing/08_02_optical/bd_delta_50yr/LC08_L2SP_136045_20180129_02_T1/red.tif` (1.8 MB) *(by file name)*
- `Data/08_remote_sensing/08_11_Capstones/bug_fire_sentinel2/bug_fire_perimeter.geojson` (368.8 KB)
- download: <https://planetarycomputer.microsoft.com/api/stac/v1>
- download: <https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/WFIGS_Interagency_Perimeters/FeatureServer/0/query>

**09_13_04_timeline_of_the_rohingya_exodus_deforestation_in_bangladesh_2016_2018.ipynb**

- `Data/08_remote_sensing/08_02_optical/bd_delta_50yr/LC08_L2SP_136045_20180129_02_T1/item.json` (36.0 KB) *(by file name)*
- `Data/08_remote_sensing/08_02_optical/bd_delta_50yr/LC08_L2SP_136045_20180129_02_T1/red.tif` (1.8 MB) *(by file name)*
- `Data/08_remote_sensing/08_02_optical/s2_rohingya_2016_2018/S2A_MSIL2A_20160105T042142_R090_T46QCJ_20210527T085101/scl.tif` (1.1 KB) *(by file name)*
- download: <https://planetarycomputer.microsoft.com/api/stac/v1>

**09_13_05_flood_mapping_with_sentinel_1_data_sylhet_bangladesh.ipynb**

- `Data/08_remote_sensing/08_03_radar/RCM1_OK2801594_PK2803067_1_SCLND_20231025_120938_VV_VH_GRD/manifest.safe` (25.6 KB) *(by file name)*
- `Data/08_remote_sensing/08_11_Capstones/Sylhet_sentinel1/*.SAFE` (0 B)
- `Data/08_remote_sensing/08_11_Capstones/Sylhet_sentinel1/sylhet_aoi.geojson` (487 B)
- `Data/08_remote_sensing/08_11_Capstones/Sylhet_sentinel1/output/sylhet_after_vh_db.tif` (14.7 MB) *(written by a notebook)*
- `Data/08_remote_sensing/08_11_Capstones/Sylhet_sentinel1/output/sylhet_before_vh_db.tif` (14.9 MB) *(written by a notebook)*
- `Data/08_remote_sensing/08_11_Capstones/Sylhet_sentinel1/output/sylhet_flood_extent.gpkg` (6.9 MB) *(written by a notebook)*
- `Data/08_remote_sensing/08_11_Capstones/Sylhet_sentinel1/output/sylhet_flood_extent.tif` (303.0 KB) *(written by a notebook)*
- `Data/08_remote_sensing/08_11_Capstones/Sylhet_sentinel1/output/sylhet_post_pre_ratio.tif` (19.7 MB) *(written by a notebook)*

**09_13_06_timeline_of_the_rohingya_exodus_deforestation_in_bangladesh_2016_2018_sentinel_1_radar.ipynb**

- `Data/08_remote_sensing/08_02_optical/bd_delta_50yr/LC08_L2SP_136045_20180129_02_T1/item.json` (36.0 KB) *(by file name)*
- `Data/08_remote_sensing/08_09_image_classification/Corn_belt/Sentinel_1/2020_05_20200517_63/vh.tif` (29.8 MB) *(by file name)*
- `Data/08_remote_sensing/08_09_image_classification/Corn_belt/Sentinel_1/2020_05_20200517_63/vv.tif` (29.8 MB) *(by file name)*
- download: <https://planetarycomputer.microsoft.com/api/stac/v1>

**09_13_07_ship_detection_in_the_strait_of_hormuz_with_sentinel_1.ipynb**

- `Data/08_remote_sensing/08_11_Capstones/hormuz_strait_sentinel1/*.SAFE` (0 B)
- `Data/08_remote_sensing/08_11_Capstones/hormuz_strait_sentinel1/AOI_hormuz_strait.geojson` (555 B)
- `Data/08_remote_sensing/08_11_Capstones/hormuz_strait_sentinel1/grd_ship_candidates.csv` (6.3 KB)
- `Data/08_remote_sensing/08_11_Capstones/hormuz_strait_sentinel1/grd_ship_candidates_abu_musa.csv` (28.3 KB)
- `Data/08_remote_sensing/08_11_Capstones/hormuz_strait_sentinel1/ne_10m_land.geojson` (9.7 MB)
- `Data/08_remote_sensing/08_11_Capstones/hormuz_strait_sentinel1/selected_grd_stac_item.json` (9.9 KB)
- `Data/08_remote_sensing/08_11_Capstones/hormuz_strait_sentinel1/S1A_IW_SLC__1SDV_20260424T020654_20260424T020721_064215_08153D_C2C2.SAFE/annotation/calibration/calibration-s1a-iw1-slc-vv-20260424t020654-20260424t020719-064215-08153d-004.xml` (922.7 KB) *(by glob pattern)*
- `Data/08_remote_sensing/08_11_Capstones/hormuz_strait_sentinel1/S1A_IW_SLC__1SDV_20260424T020654_20260424T020721_064215_08153D_C2C2.SAFE/annotation/calibration/calibration-s1a-iw2-slc-vv-20260424t020655-20260424t020720-064215-08153d-005.xml` (1.1 MB) *(by glob pattern)*
- `Data/08_remote_sensing/08_11_Capstones/hormuz_strait_sentinel1/S1A_IW_SLC__1SDV_20260424T020654_20260424T020721_064215_08153D_C2C2.SAFE/annotation/calibration/calibration-s1a-iw3-slc-vv-20260424t020656-20260424t020721-064215-08153d-006.xml` (1.0 MB) *(by glob pattern)*
- `Data/08_remote_sensing/08_11_Capstones/hormuz_strait_sentinel1/S1A_IW_SLC__1SDV_20260424T020654_20260424T020721_064215_08153D_C2C2.SAFE/annotation/calibration/noise-s1a-iw1-slc-vv-20260424t020654-20260424t020719-064215-08153d-004.xml` (125.0 KB) *(by glob pattern)*
- `Data/08_remote_sensing/08_11_Capstones/hormuz_strait_sentinel1/S1A_IW_SLC__1SDV_20260424T020654_20260424T020721_064215_08153D_C2C2.SAFE/annotation/calibration/noise-s1a-iw2-slc-vv-20260424t020655-20260424t020720-064215-08153d-005.xml` (143.3 KB) *(by glob pattern)*
- `Data/08_remote_sensing/08_11_Capstones/hormuz_strait_sentinel1/S1A_IW_SLC__1SDV_20260424T020654_20260424T020721_064215_08153D_C2C2.SAFE/annotation/calibration/noise-s1a-iw3-slc-vv-20260424t020656-20260424t020721-064215-08153d-006.xml` (139.4 KB) *(by glob pattern)*
- `Data/08_remote_sensing/08_11_Capstones/hormuz_strait_sentinel1/S1A_IW_SLC__1SDV_20260424T020654_20260424T020721_064215_08153D_C2C2.SAFE/measurement/s1a-iw1-slc-vv-20260424t020654-20260424t020719-064215-08153d-004.tiff` (1.1 GB) *(by glob pattern)*
- `Data/08_remote_sensing/08_11_Capstones/hormuz_strait_sentinel1/S1A_IW_SLC__1SDV_20260424T020654_20260424T020721_064215_08153D_C2C2.SAFE/measurement/s1a-iw2-slc-vv-20260424t020655-20260424t020720-064215-08153d-005.tiff` (1.3 GB) *(by glob pattern)*
- `Data/08_remote_sensing/08_11_Capstones/hormuz_strait_sentinel1/S1A_IW_SLC__1SDV_20260424T020654_20260424T020721_064215_08153D_C2C2.SAFE/measurement/s1a-iw3-slc-vv-20260424t020656-20260424t020721-064215-08153d-006.tiff` (1.3 GB) *(by glob pattern)*
- `Data/08_remote_sensing/08_11_Capstones/hormuz_strait_sentinel1/S1A_IW_SLC__1SDV_20260424T020719_20260424T020746_064215_08153D_650E.SAFE/annotation/calibration/calibration-s1a-iw1-slc-vv-20260424t020719-20260424t020744-064215-08153d-004.xml` (922.7 KB) *(by glob pattern)*
- `Data/08_remote_sensing/08_11_Capstones/hormuz_strait_sentinel1/S1A_IW_SLC__1SDV_20260424T020719_20260424T020746_064215_08153D_650E.SAFE/annotation/calibration/calibration-s1a-iw2-slc-vv-20260424t020720-20260424t020745-064215-08153d-005.xml` (1.1 MB) *(by glob pattern)*
- `Data/08_remote_sensing/08_11_Capstones/hormuz_strait_sentinel1/S1A_IW_SLC__1SDV_20260424T020719_20260424T020746_064215_08153D_650E.SAFE/annotation/calibration/calibration-s1a-iw3-slc-vv-20260424t020721-20260424t020746-064215-08153d-006.xml` (1.0 MB) *(by glob pattern)*
- `Data/08_remote_sensing/08_11_Capstones/hormuz_strait_sentinel1/S1A_IW_SLC__1SDV_20260424T020719_20260424T020746_064215_08153D_650E.SAFE/annotation/calibration/noise-s1a-iw1-slc-vv-20260424t020719-20260424t020744-064215-08153d-004.xml` (125.0 KB) *(by glob pattern)*
- `Data/08_remote_sensing/08_11_Capstones/hormuz_strait_sentinel1/S1A_IW_SLC__1SDV_20260424T020719_20260424T020746_064215_08153D_650E.SAFE/annotation/calibration/noise-s1a-iw2-slc-vv-20260424t020720-20260424t020745-064215-08153d-005.xml` (143.3 KB) *(by glob pattern)*
- `Data/08_remote_sensing/08_11_Capstones/hormuz_strait_sentinel1/S1A_IW_SLC__1SDV_20260424T020719_20260424T020746_064215_08153D_650E.SAFE/annotation/calibration/noise-s1a-iw3-slc-vv-20260424t020721-20260424t020746-064215-08153d-006.xml` (139.4 KB) *(by glob pattern)*
- `Data/08_remote_sensing/08_11_Capstones/hormuz_strait_sentinel1/S1A_IW_SLC__1SDV_20260424T020719_20260424T020746_064215_08153D_650E.SAFE/measurement/s1a-iw1-slc-vv-20260424t020719-20260424t020744-064215-08153d-004.tiff` (1.1 GB) *(by glob pattern)*
- `Data/08_remote_sensing/08_11_Capstones/hormuz_strait_sentinel1/S1A_IW_SLC__1SDV_20260424T020719_20260424T020746_064215_08153D_650E.SAFE/measurement/s1a-iw2-slc-vv-20260424t020720-20260424t020745-064215-08153d-005.tiff` (1.3 GB) *(by glob pattern)*
- `Data/08_remote_sensing/08_11_Capstones/hormuz_strait_sentinel1/S1A_IW_SLC__1SDV_20260424T020719_20260424T020746_064215_08153D_650E.SAFE/measurement/s1a-iw3-slc-vv-20260424t020721-20260424t020746-064215-08153d-006.tiff` (1.3 GB) *(by glob pattern)*
- `Data/08_remote_sensing/08_11_Capstones/hormuz_strait_sentinel1/S1C_IW_SLC__1SDV_20260423T021406_20260423T021436_007337_00EE1E_7D39.SAFE/annotation/calibration/calibration-s1c-iw1-slc-vv-20260423t021406-20260423t021436-007337-00ee1e-004.xml` (966.3 KB) *(by glob pattern)*
- `Data/08_remote_sensing/08_11_Capstones/hormuz_strait_sentinel1/S1C_IW_SLC__1SDV_20260423T021406_20260423T021436_007337_00EE1E_7D39.SAFE/annotation/calibration/calibration-s1c-iw2-slc-vv-20260423t021406-20260423t021434-007337-00ee1e-005.xml` (1.1 MB) *(by glob pattern)*
- `Data/08_remote_sensing/08_11_Capstones/hormuz_strait_sentinel1/S1C_IW_SLC__1SDV_20260423T021406_20260423T021436_007337_00EE1E_7D39.SAFE/annotation/calibration/calibration-s1c-iw3-slc-vv-20260423t021407-20260423t021435-007337-00ee1e-006.xml` (1.1 MB) *(by glob pattern)*
- `Data/08_remote_sensing/08_11_Capstones/hormuz_strait_sentinel1/S1C_IW_SLC__1SDV_20260423T021406_20260423T021436_007337_00EE1E_7D39.SAFE/annotation/calibration/noise-s1c-iw1-slc-vv-20260423t021406-20260423t021436-007337-00ee1e-004.xml` (144.6 KB) *(by glob pattern)*
- `Data/08_remote_sensing/08_11_Capstones/hormuz_strait_sentinel1/S1C_IW_SLC__1SDV_20260423T021406_20260423T021436_007337_00EE1E_7D39.SAFE/annotation/calibration/noise-s1c-iw2-slc-vv-20260423t021406-20260423t021434-007337-00ee1e-005.xml` (153.0 KB) *(by glob pattern)*
- `Data/08_remote_sensing/08_11_Capstones/hormuz_strait_sentinel1/S1C_IW_SLC__1SDV_20260423T021406_20260423T021436_007337_00EE1E_7D39.SAFE/annotation/calibration/noise-s1c-iw3-slc-vv-20260423t021407-20260423t021435-007337-00ee1e-006.xml` (149.0 KB) *(by glob pattern)*
- `Data/08_remote_sensing/08_11_Capstones/hormuz_strait_sentinel1/S1C_IW_SLC__1SDV_20260423T021406_20260423T021436_007337_00EE1E_7D39.SAFE/measurement/s1c-iw1-slc-vv-20260423t021406-20260423t021436-007337-00ee1e-004.tiff` (1.3 GB) *(by glob pattern)*
- `Data/08_remote_sensing/08_11_Capstones/hormuz_strait_sentinel1/S1C_IW_SLC__1SDV_20260423T021406_20260423T021436_007337_00EE1E_7D39.SAFE/measurement/s1c-iw2-slc-vv-20260423t021406-20260423t021434-007337-00ee1e-005.tiff` (1.4 GB) *(by glob pattern)*
- `Data/08_remote_sensing/08_11_Capstones/hormuz_strait_sentinel1/S1C_IW_SLC__1SDV_20260423T021406_20260423T021436_007337_00EE1E_7D39.SAFE/measurement/s1c-iw3-slc-vv-20260423t021407-20260423t021435-007337-00ee1e-006.tiff` (1.3 GB) *(by glob pattern)*
- `Data/08_remote_sensing/08_11_Capstones/hormuz_strait_sentinel1/S1C_IW_SLC__1SDV_20260423T021434_20260423T021501_007337_00EE1E_6B31.SAFE/measurement/s1c-iw1-slc-vv-20260423t021436-20260423t021501-007337-00ee1e-004.tiff` (1.0 GB) *(by glob pattern)*
- `Data/08_remote_sensing/08_11_Capstones/hormuz_strait_sentinel1/S1C_IW_SLC__1SDV_20260423T021434_20260423T021501_007337_00EE1E_6B31.SAFE/measurement/s1c-iw2-slc-vv-20260423t021434-20260423t021459-007337-00ee1e-005.tiff` (1.2 GB) *(by glob pattern)*
- `Data/08_remote_sensing/08_11_Capstones/hormuz_strait_sentinel1/S1C_IW_SLC__1SDV_20260423T021434_20260423T021501_007337_00EE1E_6B31.SAFE/measurement/s1c-iw3-slc-vv-20260423t021435-20260423t021500-007337-00ee1e-006.tiff` (1.2 GB) *(by glob pattern)*
- download: <https://planetarycomputer.microsoft.com/api/stac/v1/search>
- download: <https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-1-grd>
- download: <https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_land.geojson>

**09_13_08_crop_classification_with_nisar_gcov_time_series_and_self_supervised_learning.ipynb**

- `Data/08_remote_sensing/08_11_Capstones/NISAR_crop_classification/2025_10m_cdls/2025_10m_cdls.tif` (13.9 GB)
- `Data/08_remote_sensing/08_11_Capstones/NISAR_crop_classification/NISAR_2026/NISAR_L2_*GCOV*.h5` (0 B)
- `Data/08_remote_sensing/08_11_Capstones/NISAR_crop_classification/outputs_ssl_crop/`: 24 files (.csv, .json, .png, .pt, .tif), 23.4 MB *(written by a notebook)*

**09_13_09_sea_ice_mapping_for_shipping_routes_in_the_st_lawrence_estuary_with_radarsat_constellation_mission_rcm.ipynb**

- `Data/08_remote_sensing/08_03_radar/RCM1_OK2801594_PK2803067_1_SCLND_20231025_120938_VV_VH_GRD/metadata/product.xml` (2.4 MB) *(by glob pattern)*
- `Data/08_remote_sensing/08_03_radar/RS2_OK153952_PK1408399_DK1372518_FQ17_20150626_052324_HH_VV_HV_VH_SLC/product.xml` (111.2 KB) *(by glob pattern)*
- `Data/08_remote_sensing/08_03_radar/RS2_OK153952_PK1408400_DK1372519_XF0W2_20210705_052511_HH_SLC/product.xml` (396.7 KB) *(by glob pattern)*
- `Data/08_remote_sensing/08_03_radar/RS2_OK153952_PK1408401_DK1372520_MF4N_20231113_051222_HH_SGF/product.xml` (72.6 KB) *(by glob pattern)*
- `Data/08_remote_sensing/08_03_radar/RS2_OK153952_PK1408402_DK1372521_S3_20231116_052449_HH_HV_SGF/product.xml` (72.9 KB) *(by glob pattern)*
- `Data/08_remote_sensing/08_03_radar/RS2_OK153952_PK1408403_DK1372522_SQ1_20231119_053723_HH_VV_HV_VH_SLC/product.xml` (88.0 KB) *(by glob pattern)*
- `Data/08_remote_sensing/08_03_radar/RS2_OK153952_PK1408404_DK1372523_F21F_20231123_052042_HH_SGF/product.xml` (69.6 KB) *(by glob pattern)*
- `Data/08_remote_sensing/08_03_radar/RS2_OK153952_PK1408405_DK1372524_W3_20231130_051625_HH_HV_SGF/product.xml` (72.9 KB) *(by glob pattern)*
- `Data/08_remote_sensing/08_03_radar/rcm_insar_montreal/RCM1_OK2678697_PK2744956_1_16M11_20230913_224404_VV_SLC/metadata/product.xml` (167.1 KB) *(by glob pattern)*
- `Data/08_remote_sensing/08_03_radar/rcm_insar_montreal/RCM2_OK2678697_PK2750939_1_16M11_20230917_224416_VV_SLC/metadata/product.xml` (166.2 KB) *(by glob pattern)*
- `Data/08_remote_sensing/08_03_radar/s1_rohingya_2017/RS2_OK153952_PK1408402_DK1372521_S3_20231116_052449_HH_HV_SGF/product.xml` (72.9 KB) *(by glob pattern)*
- `Data/08_remote_sensing/08_11_Capstones/RADARSAT_ICE/aoi.kml` (596 B)
- `Data/08_remote_sensing/08_11_Capstones/RADARSAT_ICE/RCM1_OK3695251_PK4015454_1_SCLNA_20260128_104701_HH_HV_GRD/RCM1_OK3695251_PK4015454_1_SCLNA_20260128_104701_HH_HV_GRD/metadata/product.xml` (935.5 KB) *(by glob pattern)*
- `Data/08_remote_sensing/08_11_Capstones/RADARSAT_ICE/RCM3_OK3695270_PK4023614_1_SCLND_20260203_103123_HH_HV_GRD/RCM3_OK3695270_PK4023614_1_SCLND_20260203_103123_HH_HV_GRD/metadata/product.xml` (730.6 KB) *(by glob pattern)*
- `Data/08_remote_sensing/08_11_Capstones/hormuz_strait_sentinel1/ne_10m_land.geojson` (9.7 MB)
- download: <https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_land.geojson>
- download: <http://www.opengis.net/kml/2.2>
- download: <https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}>

**09_13_10_charland_monitoring_with_sentinel_1_sar_and_nisar_l_band_lower_meghna_estuary_of_bangladesh.ipynb**

- `Data/00_shared/data_web_cloud_gis/08_02_outputs/unions.gpkg` (123.8 MB) *(by file name)*
- `Data/08_remote_sensing/08_03_radar/RCM1_OK2801594_PK2803067_1_SCLND_20231025_120938_VV_VH_GRD/manifest.safe` (25.6 KB) *(by file name)*
- `Data/08_remote_sensing/08_11_Capstones/s1_BD_charland/charland_aoi.geojson` (718 B)
- `Data/08_remote_sensing/08_11_Capstones/s1_BD_charland/scene_catalog.csv` (3.8 KB)
- `Data/08_remote_sensing/08_11_Capstones/s1_BD_charland/NISAR_BD/NISAR_L2_*GCOV*.h5` (0 B)
- `Data/08_remote_sensing/08_11_Capstones/s1_BD_charland/Sentinel_1/*.SAFE` (0 B)
- `Data/08_remote_sensing/08_11_Capstones/s1_BD_charland/annual/first_emergence_year.tif` (1.0 MB) *(by file name)*
- `Data/08_remote_sensing/08_11_Capstones/s1_BD_charland/annual/loss_year.tif` (1.2 MB) *(by file name)*
- `Data/08_remote_sensing/08_11_Capstones/s1_BD_charland/validation/ref_points_single_date.gpkg` (212.0 KB) *(by file name)*
- `Data/08_remote_sensing/08_11_Capstones/s1_BD_charland/vectors/char_register.csv` (178.0 KB) *(by file name)*
- `Data/08_remote_sensing/08_11_Capstones/s1_BD_charland/vectors/transects.gpkg` (9.1 MB) *(by file name)*

**09_13_11_mapping_shrimp_ghers_in_coastal_bangladesh_with_a_2025_2026_sentinel_1_time_series_and_sentinel_2.ipynb**

- `Data/08_remote_sensing/08_11_Capstones/SAR_BD_Coastal_District/AOI.kml` (1018 B) *(by file name)*
- `Data/08_remote_sensing/08_11_Capstones/SAR_BD_Coastal_District/khulna_satkhira_bagerhat.geojson` (172.0 KB) *(by file name)*
- `Data/08_remote_sensing/08_11_Capstones/SAR_BD_Coastal_District/cache/jrc_occurrence_20m_gher.nc` (47.0 MB) *(by file name)*
- `Data/08_remote_sensing/08_11_Capstones/SAR_BD_Coastal_District/cache/worldcover2021_20m_gher.nc` (11.8 MB) *(by file name)*
- download: <https://planetarycomputer.microsoft.com/api/stac/v1>

**09_13_12_mapping_the_2026_big_sur_wildfire_with_time_series_sentinel_1_sar_and_sentinel_2_optical_images.ipynb**

- `Data/08_remote_sensing/08_11_Capstones/S1_Rice_Bnagladesh/cache/s2_region_100m/S2_202512.nc` (259.9 MB) *(by glob pattern)*
- `Data/08_remote_sensing/08_11_Capstones/S1_Rice_Bnagladesh/cache/s2_region_100m/S2_202601.nc` (259.9 MB) *(by glob pattern)*
- `Data/08_remote_sensing/08_11_Capstones/S1_Rice_Bnagladesh/cache/s2_region_100m/S2_202602.nc` (259.9 MB) *(by glob pattern)*
- `Data/08_remote_sensing/08_11_Capstones/S1_Rice_Bnagladesh/cache/s2_region_100m/S2_202603.nc` (259.9 MB) *(by glob pattern)*
- `Data/08_remote_sensing/08_11_Capstones/s1_Big _Sur_wildfires/AOI_Big_Sur.geojson` (577 B) *(by file name)*
- `Data/08_remote_sensing/08_11_Capstones/s1_Big _Sur_wildfires/cache/copdem30_10m.nc` (7.9 MB) *(by file name)*
- `Data/08_remote_sensing/08_11_Capstones/s1_Big _Sur_wildfires/cache/s1_rtc_10m/`: 36 files (.nc), 673.8 MB
- `Data/08_remote_sensing/08_11_Capstones/s1_Big _Sur_wildfires/cache/s2_l2a_10m/`: 25 files (.nc), 1.4 GB
- download: <https://planetarycomputer.microsoft.com/api/stac/v1>

**09_13_13_mapping_wetland_paddy_in_bangladesh_with_a_2025_to_2026_sentinel_1_time_series.ipynb**

- `Data/08_remote_sensing/08_09_image_classification/S1_Rice_Bnagladesh/AOI_1.kml` (1018 B) *(by file name)*
- `Data/08_remote_sensing/08_09_image_classification/S1_Rice_Bnagladesh/cache/geoBoundaries-BGD-ADM2_simplified.geojson` (1.6 MB) *(by file name)*
- `Data/08_remote_sensing/08_09_image_classification/S1_Rice_Bnagladesh/cache/jrc_occurrence_20m_haor.nc` (18.9 MB) *(by file name)*
- `Data/08_remote_sensing/08_09_image_classification/S1_Rice_Bnagladesh/cache/worldcover2021_20m_haor.nc` (4.8 MB) *(by file name)*
- download: <https://planetarycomputer.microsoft.com/api/stac/v1>

**09_13_15_tree_inventory_biomass_and_carbon_from_airborne_lidar.ipynb**

- `Data/08_remote_sensing/08_02_optical/ecostress_lst.tif` (3.4 MB) *(by file name)*
- `Data/08_remote_sensing/08_03_radar/dem_90m.tif` (17.1 MB) *(by file name)*
- `Data/08_remote_sensing/08_03_radar/landsat.tif` (9.7 MB) *(by file name)*
- `Data/08_remote_sensing/08_06_lidar/madison.copc.laz` (22.1 MB) *(by file name)*
- download: <https://github.com/opengeos/datasets/releases/download>
- download: <https://raw.githubusercontent.com>

**09_13_16_uav_based_high_throughput_phenotyping_of_maize_nitrogen_response.ipynb**

- `Data/08_remote_sensing/08_11_Capstones/maize_uav_htp/field/2019_yield_traits.csv` (92.3 KB)
- `Data/08_remote_sensing/08_11_Capstones/maize_uav_htp/field/CRRI_2019_Field_Book_final.csv` (67.5 KB)
- `Data/08_remote_sensing/08_11_Capstones/maize_uav_htp/field/leaf_traits_spectra_2019.csv` (12.2 MB)
- `Data/08_remote_sensing/08_11_Capstones/maize_uav_htp/genotypes/genotype_to_vcf_id.csv` (5.4 KB)
- `Data/08_remote_sensing/08_11_Capstones/maize_uav_htp/genotypes/results/*.top.parquet` (0 B)
- `Data/08_remote_sensing/08_11_Capstones/maize_uav_htp/genotypes/results/gwas_meta.json` (137 B)
- `Data/08_remote_sensing/08_11_Capstones/maize_uav_htp/genotypes/results/gwas_regions.csv` (2.2 KB)
- `Data/08_remote_sensing/08_11_Capstones/maize_uav_htp/genotypes/results/gwas_summary.csv` (1.1 KB)
- `Data/08_remote_sensing/08_11_Capstones/maize_uav_htp/genotypes/results/kinship_cXX.txt` (665.2 KB)
- `Data/08_remote_sensing/08_11_Capstones/maize_uav_htp/genotypes/results/kinship_ids.txt` (2.7 KB)
- `Data/08_remote_sensing/08_11_Capstones/maize_uav_htp/genotypes/results_emmax/*.top.parquet` (0 B)
- `Data/08_remote_sensing/08_11_Capstones/maize_uav_htp/genotypes/results_emmax/emmax_regions.csv` (17.2 KB)
- `Data/08_remote_sensing/08_11_Capstones/maize_uav_htp/genotypes/results_emmax/emmax_summary.csv` (18.8 KB)
- `Data/08_remote_sensing/08_11_Capstones/maize_uav_htp/gwas/Table_S3_GWAS_regions.csv` (23.2 KB)
- `Data/08_remote_sensing/08_11_Capstones/maize_uav_htp/outputs/*` (0 B) *(written by a notebook)*
- `Data/08_remote_sensing/08_11_Capstones/maize_uav_htp/outputs/manifest.txt` (948 B) *(written by a notebook)*
- `Data/08_remote_sensing/08_11_Capstones/maize_uav_htp/outputs/figures/`: 14 files (.png), 8.3 MB *(written by a notebook)*
- `Data/08_remote_sensing/08_11_Capstones/maize_uav_htp/outputs/tables/`: 15 files (.csv, .json), 4.8 MB *(written by a notebook)*
- `Data/08_remote_sensing/08_11_Capstones/maize_uav_htp/plot_traits/CC_pval_checks_ExG131.csv` (1.2 KB)
- `Data/08_remote_sensing/08_11_Capstones/maize_uav_htp/plot_traits/CC_pval_genotype_ExG131.csv` (1000 B)
- `Data/08_remote_sensing/08_11_Capstones/maize_uav_htp/plot_traits/VI_average_fb2019_n46364_ref.csv` (16.9 MB)
- `Data/08_remote_sensing/08_11_Capstones/maize_uav_htp/plot_traits/check_ref.csv` (1.7 MB)
- `Data/08_remote_sensing/08_11_Capstones/maize_uav_htp/plot_traits/geno_ref.csv` (11.4 MB)
- `Data/08_remote_sensing/08_11_Capstones/maize_uav_htp/uav_images/DJI_0004.JPG` (7.8 MB)
- `Data/08_remote_sensing/08_11_Capstones/maize_uav_htp/uav_images/DJI_0009.JPG` (7.8 MB)
- `Data/08_remote_sensing/08_11_Capstones/maize_uav_htp/uav_images/havelock_2018.jpg` (342.2 KB)

## Datasets shared between chapters

| Dataset folder | Chapters | Notebooks | Files |
|---|---|---:|---:|
| `Data/00_shared/vector_data` | 01_vector, 02_raster, 03_spatial_analysis, 06_spatialDB, 07_web_cloud_gis, 08_remote_sensing | 29 | 46 |
| `Data/00_shared/raster_data` | 02_raster, 03_spatial_analysis, 06_spatialDB, 08_remote_sensing | 14 | 135 |
| `Data/00_shared/data_geostatistics` | 05_geostatistics, 06_spatialDB, 08_remote_sensing | 53 | 92 |
| `Data/00_shared/data_remote_sensing` | 06_spatialDB, 07_web_cloud_gis, 08_remote_sensing | 17 | 52 |
| `Data/00_shared/data_web_cloud_gis` | 06_spatialDB, 07_web_cloud_gis, 08_remote_sensing | 13 | 86 |
| `Data/00_shared/tabular_data` | 01_vector, 03_spatial_analysis, 07_web_cloud_gis | 4 | 4 |
| `Data/00_shared/data_spatial_stat` | 04_spatial_statistics, 07_web_cloud_gis | 75 | 18 |
| `Data/00_shared/data_spatial_net` | 03_spatial_analysis, 06_spatialDB | 7 | 9 |
| `Data/03_spatial_analysis/Great_plain` | 03_spatial_analysis, 08_remote_sensing | 3 | 7 |
| `Data/00_shared/data_gdal` | 01_vector, 06_spatialDB | 2 | 12 |
