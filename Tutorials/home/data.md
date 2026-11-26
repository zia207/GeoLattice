---
title: "Data"
subtitle: "The sample data used by the GeoLattice tutorials"
---

The tutorial notebooks work on real data: county boundaries and soil samples, DEMs and climate grids, NYC taxi trips, and Landsat, Sentinel, RADARSAT, NISAR, hyperspectral, LiDAR and UAV imagery. Together the data come to about **540 GB**, far more than GitHub can hold, so they are kept in a shared Dropbox folder, `GeoLattice_Data`.

::: {.callout-note appearance="simple" icon="false"}
### [Open the GeoLattice data folder on Dropbox](https://www.dropbox.com/scl/fo/vt4g2g7tpvue8c4xpiv9z/AJcsXd452txjqB3vT8pS9Zw?rlkey=dtxhrf9ti74bh3bjly2uw9623&st=8z7qfknu&dl=0){.btn .btn-primary target="_blank"}

No Dropbox account is needed to browse or download. With an account you can also choose **Copy to Dropbox** and sync only the folders you need.
:::

## Setting up the data

The notebooks expect the data in a folder named `Data` at the root of the GeoLattice repository:

```text
GeoLattice/
├── Data/                      ← the Dropbox folder GeoLattice_Data, renamed
│   ├── 00_shared/
│   ├── 03_spatial_analysis/
│   ├── 06_spatialDB/
│   ├── 08_distributed_spatial/
│   ├── 08_remote_sensing/
│   ├── 00_unassigned/
│   ├── MOVE_MANIFEST.csv
│   └── README.md
├── Tutorials/
└── src/
```

1. Clone the repository: `git clone https://github.com/zia207/GeoLattice.git`.
2. Download the folders you need from Dropbox (see [Which folders each chapter needs](#which-folders-each-chapter-needs)) and unzip them into `GeoLattice/Data/`, keeping the folder names and nesting exactly as they are on Dropbox.
3. Open a notebook from `Tutorials/notebooks/` and run it. Notebooks find the data from their own location, so nothing else needs to be configured.

You rarely need everything. Most chapters need a few gigabytes or less; the largest share is remote sensing imagery, where each section can be downloaded on its own. Dropbox may refuse to zip a very large folder in one go; download its subfolders one at a time instead.

Some notebooks also download data at run time (STAC catalogues such as Microsoft Planetary Computer, NASA Earthdata, ASF, Copernicus, OpenStreetMap and public APIs) and cache it under `Data/`. NASA Earthdata downloads need a free [Earthdata Login](https://urs.earthdata.nasa.gov/) saved in `~/.netrc`; a notebook says so where it needs one.

## What is in the folder

| Folder | Size | Contents | Used by chapters |
|---|---:|---|---|
| `00_shared/` | 127 GB | Datasets read by notebooks in more than one chapter (below) | 1 to 9 |
| `03_spatial_analysis/` | 18 GB | Study areas for spatial analysis: Great Plains, Sylhet floods, wheat suitability, musk deer habitat, NetCDF climate cubes, segregation and causal data | 3, 9 |
| `06_spatialDB/` | 12 GB | NYC yellow taxi trips (2023) in CSV, Parquet, Feather and TSV, taxi zones, Natural Earth, OpenStreetMap extracts and the city atlas | 6 |
| `08_distributed_spatial/` | 184 MB | Kansas building footprints (partitioned GeoParquet) and Wichita roads and facilities | 8 |
| `08_remote_sensing/` | 388 GB | Imagery for the remote sensing chapter, one subfolder per section (below) | 9 |
| `00_unassigned/` | 209 MB | Archives and folders no notebook reads; not needed to run the tutorials | – |
| `README.md`, `MOVE_MANIFEST.csv` | | Folder guide and a record of where each dataset came from in earlier versions of the repository | |

### Shared datasets, `00_shared/`

| Folder | Size | Contents | Used by chapters |
|---|---:|---|---|
| `vector_data/` | 640 MB | Bangladesh boundaries (country, divisions, districts, upazilas) with soil, arsenic, rice and poverty tables; US states and counties, New York and Colorado layers, Onondaga streets and Yellowstone | 1, 2, 3, 6, 7, 9 |
| `raster_data/` | 13 GB | DEMs, NDVI and water table rasters for Bangladesh; elevation of Colorado and the Great Plains and DEM tiles for the United States; GDAL sample rasters | 2, 3, 6, 9 |
| `data_gdal/` | 26 MB | Small test files in about 40 vector formats (GeoJSON, GML, KML, GeoPackage, FlatGeobuf, File Geodatabase, OSM, GPX, DXF and more) | 1, 6 |
| `tabular_data/` | 10 MB | Tables to join to maps: Bangladesh poverty, rice production and soils, BGS groundwater geochemistry, US diabetes, climate | 1, 3, 7 |
| `data_spatial_net/` | 1 MB | A synthetic study area with roads, towns, hospitals, farmers, census tracts and commuting flows | 3, 6 |
| `data_spatial_stat/` | 575 MB | US county boundaries and health data, California PM2.5, NYC crime incidents, Buffalo–Niagara census tracts (1990 to 2020) and panel data | 4, 7 |
| `data_geostatistics/` | 1.5 GB | Point samples for kriging: Bangladesh groundwater arsenic and BGS geochemistry, Great Plains soil samples and prediction grids, California PM2.5, US diabetes and cancer data, GSLIB examples | 5, 6, 9 |
| `data_web_cloud_gis/` | 1.7 GB | Sentinel-2 over Syracuse (2025) and the outputs of the web and cloud GIS notebooks | 6, 7, 9 |
| `data_remote_sensing/` | 110 GB | Landsat 8 and 9 scenes (Bangladesh, Las Vegas), Sentinel-1, Sentinel-2, MODIS, commercial images, LiDAR, the Ithaca data and UAV imagery | 6, 7, 9 |

### Remote sensing, `08_remote_sensing/`

The folders still carry the remote sensing chapter's earlier number, 08; folder `08_02_optical` holds the data of tutorial section 09_02, and so on.

| Folder | Section | Size | Contents |
|---|---|---:|---|
| `08_02_optical/` | 9.2 Optical | 9 GB | Landsat time series (Pucallpa 1985 to 2020, Bangladesh delta over 50 years), MODIS, Sentinel-3, ECOSTRESS, NLCD, biophysical parameters |
| `08_03_radar/` | 9.3 Radar | 67 GB | Sentinel-1 GRD and time series, RADARSAT-2 and RCM scenes, NISAR over Bangladesh, InSAR (Mexico City, Death Valley, Montreal), Copernicus DEM |
| `08_04_uav/` | 9.4 UAV | 31 GB | UAV field trials (canola, dry bean, lentil and wheat), Bangladesh flights and infrastructure-free phenotyping |
| `08_05_HyperSpectral/` | 9.5 Hyperspectral | 18 GB | Tanager scenes, Pavia University, a 10 m sample scene and HITRAN absorption data |
| `08_06_lidar/` | 9.6 LiDAR | 5 GB | Airborne point clouds (Ithaca, Madison, Autzen and others) and simulated ICESat-2 and GEDI granules |
| `08_08_image_super_resolution/` | 9.8 Super-resolution | 6 GB | Matched Landsat 8, Sentinel-2, RapidEye, Planet and high-resolution images |
| `08_09_image_classification/` | 9.9 Image classification | 21 GB | Corn Belt crop maps and imagery, WorldView-2, Sentinel-1 and Sentinel-2 scenes, land cover samples, image fusion |
| `08_10_object_detection/` | 9.10 Object detection | 3 GB | Pool, tree and field-edge detection images and labels |
| `08_11_Capstones/` | 9.14 Capstones | 230 GB | The capstone projects: Sentinel-1 and Sentinel-2 time series (Rohingya camps, Sylhet floods, Big Sur fire, charlands, shrimp ghers, paddy), Strait of Hormuz, NISAR crops, RADARSAT sea ice, field edges, UAV maize phenotyping |
| `08_12_surface_energy_balance/` | 9.12 Surface energy balance | 0.9 GB | Landsat stacks and weather for Mymensingh (Bangladesh) and California's Westside |
| `08_13_soil_moisture/` | 9.13 Soil moisture | 1.6 GB | SMAP, SMOS, ESA CCI, ERA5-Land, NISAR, Sentinel-1, MODIS, gridMET, SoilGrids, SMAPVEX16, SPL2SMAP_S and in situ stations |
| `outputs/` | | 0.3 GB | Rasters written by the LiDAR notebooks, kept so later notebooks can reuse them |

## Which folders each chapter needs

| Chapter | Folders to download | About |
|---|---|---:|
| [1. Vector Data](notebooks/01_02_vector_raster/index.html#vector-data) | `00_shared/vector_data`, `00_shared/data_gdal`, `00_shared/tabular_data` | 0.7 GB |
| [2. Raster Data](notebooks/01_02_vector_raster/index.html#raster-data) | `00_shared/raster_data`, `00_shared/vector_data` | 14 GB |
| [3. Spatial Analysis](notebooks/03_spatial_analysis/index.html) | `03_spatial_analysis`, `00_shared/raster_data`, `00_shared/vector_data`, `00_shared/data_spatial_net`, `00_shared/tabular_data` | 32 GB |
| [4. Spatial Statistics](notebooks/04_spatial_statistics/index.html) | `00_shared/data_spatial_stat` | 0.6 GB |
| [5. Geostatistics](notebooks/05_geostatistics/index.html) | `00_shared/data_geostatistics` | 1.5 GB |
| [6. Spatial Database](notebooks/06_spatialDB/index.html) | `06_spatialDB`, plus single files from `00_shared` (`vector_data`, `raster_data`, `data_gdal`, `data_geostatistics`, `data_spatial_net`, `data_web_cloud_gis`) | 12 GB |
| [7. Web & Cloud GIS](notebooks/07_web_cloud_gis/index.html) | `00_shared/data_web_cloud_gis`, `00_shared/vector_data`, `00_shared/tabular_data`, `00_shared/data_spatial_stat`, parts of `00_shared/data_remote_sensing` | 3 GB and up |
| [8. Distributed Spatial Analysis](notebooks/08_distributed_spatial/index.html) | `08_distributed_spatial`, `03_spatial_analysis/Great_plain`, `00_shared/data_geostatistics/Data_GP`, `06_spatialDB/yellow_tripdata_2023.parquet` and `taxi_zones`, the 2025 Cropland Data Layer in `08_remote_sensing/08_11_Capstones/NISAR_crop_classification/2025_10m_cdls` | 32 GB |
| [9. Remote Sensing](notebooks/09_remote_sensing/index.html) | The `08_remote_sensing` subfolder of the section you are working on, plus `00_shared/data_remote_sensing` | 1 to 230 GB per section |

For the exact files each notebook reads, see [DATA_BY_CHAPTER.md](https://github.com/zia207/GeoLattice/blob/main/Tutorials/DATA_BY_CHAPTER.md) in the repository (also as a spreadsheet, [DATA_BY_CHAPTER.xlsx](https://github.com/zia207/GeoLattice/blob/main/Tutorials/DATA_BY_CHAPTER.xlsx), one sheet per chapter).

## Terms of use

The datasets come from public sources such as the US Census Bureau, USGS, USDA, NASA, ESA and Copernicus, Natural Earth, OpenStreetMap and the NYC Taxi and Limousine Commission, and are shared here for learning with these tutorials. Each source's own license and citation requirements still apply, and commercial images (for example WorldView, Planet, RapidEye and Tanager) remain under their providers' licenses. The notebooks name the source of each dataset they use.
