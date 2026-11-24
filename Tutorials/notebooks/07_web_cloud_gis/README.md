# Module 08: Web and Cloud GIS with GeoLattice

Take a GeoLattice analysis off the laptop: read data straight from cloud storage, process it at
scale, publish it as maps, tiles and APIs, and deploy small web applications.

**Level:** intermediate. Assumes Modules 01–02 (vector and raster) and 07_01_04 (data discovery).

**Main tools:** `geolattice.web` (cloud access, COG, tiles, tile and OGC API server),
`geolattice.vector` (GeoParquet, FlatGeobuf, PMTiles), `geolattice.rsense.get_images` (search and
download imagery from Earth Search, Planetary Computer, Copernicus, NOAA and USGS),
`geolattice.rsense` (reading, indices, Zarr, STAC). Install with `pip install "geolattice[web,rsense]"`.

**Data:** the GeoLattice `data/` folder (`vector_data/USA`, `data_remote_sensing`, `raster_data`),
plus Sentinel-2 and Landsat scenes downloaded with `get_images` into
`Data/00_shared/data_web_cloud_gis/`, which later notebooks reuse.

| Notebook | Topic | Status |
|---|---|---|
| `07_00_introduction_web_cloud_gis.ipynb` | Web vs cloud GIS, architecture, standards, range requests, tiles, catalogues | ready |
| `07_01_cloud_native_raster.ipynb` | COG, Zarr and STAC: creating, validating, reading remotely, cataloguing | ready |
| `07_02_cloud_native_vector.ipynb` | GeoParquet, FlatGeobuf, PMTiles, GeoJSON sequences; measured remote reads | ready |
| `07_03_spatial_sql.ipynb` | DuckDB spatial SQL, join quality, GeoPackage SQL, PostGIS and cloud SQL | ready |
| `07_04_interactive_web_maps.ipynb` | Folium, leafmap, lonboard; preparing vectors (coverage simplification), choropleth classes, many points, rasters as overlays and tiles | ready |
| `07_05_serving_maps.ipynb` | Static vs dynamic tiles, COG bytes per tile, OGC API – Features (paging, bbox, datetime) tested with GDAL/OWSLib, worker processes, PMTiles + MapLibre | ready |
| `07_06_geospatial_apis.ipynb` | FastAPI: validation, spatial and raster endpoints, background jobs, keys, rate limits, gzip, OGC mount, async vs def, tests | ready |
| `07_07_web_apps.ipynb` | One district explorer in Streamlit, Panel and Solara (apps/ folder): shared logic module, precomputed data, headless tests, execution models, servers | ready |
| `07_08_scalable_processing.ipynb` | National NDVI trend (6.5 billion slopes): parallel reads, Dask pipelines and scaling, Zarr chunking, CuPy GPU, remote COG throughput, batch jobs and cost | ready |
| `07_09_cloud_platforms.ipynb` | Three STAC catalogues compared, the Sentinel-2 offset across providers, odc-stac composite, global layers for Sunamganj, openEO graph, Earth Engine (needs account) | ready |
| `07_10_devops_geospatial.ipynb` | The `haor_monitor/` project: package, tests and coverage gates, a broken change caught, git and CI workflow, Dockerfile checks, idempotent scheduled runs, JSON logs | ready |
| `07_11_capstone.ipynb` | Haor flood monitor: Sentinel-1 water every 12 days (VH vs wind-sensitive VV), union statistics, validation against Sentinel-2 and JRC, COG + GeoParquet + STAC, FastAPI + OGC service + web map, tests | ready |
