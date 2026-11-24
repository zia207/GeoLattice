# haor-monitor

Scheduled Sentinel-2 monitoring of open water and vegetation in Tanguar Haor (Sunamganj,
Bangladesh), built in GeoLattice Notebook 08_10 as an example of a production-style geospatial pipeline.

For every new Sentinel-2 overpass the job mosaics the scenes over the haor, masks clouds with the
scene classification layer, and records the clear fraction, the open-water fraction (NDWI > 0) and
the median NDVI of land. Results are appended to `observations.parquet`; every run is logged as JSON
lines in `runs.jsonl`. The job is idempotent: already processed dates are skipped.

```bash
pip install -e ".[dev]"
ruff check . && pytest
python -m haor_monitor run --start 2024-01-01 --end 2024-12-31
python -m haor_monitor report
```

Configuration comes from environment variables: `HAOR_DATA_DIR`, `HAOR_AOI` (west,south,east,north),
`HAOR_MAX_CLOUD`, `HAOR_RESOLUTION`, `HAOR_STAC_URL`, `HAOR_COLLECTION`, `HAOR_MIN_CLEAR`.

Container: `docker build -t haor-monitor .` then
`docker run --rm -v "$PWD/monitor_data:/data" haor-monitor run --start 2024-01-01`.
CI (`.github/workflows/ci.yml`) lints, tests and builds the image; `monitor.yml` runs the job every
five days and commits the results to a `data` branch.
