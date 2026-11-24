"""
Build the USA Health Data Atlas SQLite database used by the voting-pattern capstone.

Steps
1. Create the schema and dataset catalog in
   Data/00_shared/data_spatial_stat/usa_health_atlas_database/usa_health_atlas.sqlite
   (and write schema.sql / README.md next to it).
2. Download the Atlas JavaScript data bundles and import them with the
   dataset-specific parsers in build_database.py: presidential returns
   2004-2024, Census PEP demography, CDC/ATSDR SVI 2022, U.S. Religion Census
   2010/2020, and County Health Rankings 2020-2025.

Usage (from anywhere in the repository):
    python Tutorials/notebooks/04_spatial_statistics/04_10_capstone/data_processing.py
    python .../data_processing.py --schema-only        # step 1 only
    python .../data_processing.py --assets-dir /tmp/atlas_assets

Re-running is safe: each dataset's rows are replaced, not duplicated.
"""
import argparse
import importlib.util
import sqlite3
from pathlib import Path


def repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").is_file() and (candidate / "Data").is_dir():
            return candidate
    raise FileNotFoundError("Could not find the GeoLattice repository root.")


ROOT = repo_root(Path(__file__).resolve().parent)
out = ROOT / "Data" / "00_shared" / "data_spatial_stat" / "usa_health_atlas_database"
out.mkdir(parents=True, exist_ok=True)
db_path = out / "usa_health_atlas.sqlite"

schema = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS dataset_catalog (
    dataset_id INTEGER PRIMARY KEY,
    dataset_name TEXT NOT NULL UNIQUE,
    source_page TEXT,
    source_provider TEXT,
    source_url TEXT,
    geography_level TEXT,
    temporal_coverage TEXT,
    notes TEXT,
    loaded_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS counties (
    fips TEXT PRIMARY KEY CHECK(length(fips) = 5),
    county_name TEXT,
    state_name TEXT,
    state_abbr TEXT,
    state_fips TEXT,
    county_fips TEXT,
    latitude REAL,
    longitude REAL,
    centroid_x_m REAL,
    centroid_y_m REAL
);

-- Flexible long-form storage: one row per dataset observation / variable.
CREATE TABLE IF NOT EXISTS county_indicators (
    observation_id INTEGER PRIMARY KEY,
    dataset_id INTEGER NOT NULL REFERENCES dataset_catalog(dataset_id),
    fips TEXT NOT NULL,
    year INTEGER,
    variable TEXT NOT NULL,
    value REAL,
    unit TEXT,
    source_geography TEXT,
    source_record_id TEXT,
    FOREIGN KEY (fips) REFERENCES counties(fips)
);

CREATE INDEX IF NOT EXISTS idx_indicators_fips_year
    ON county_indicators(fips, year);
CREATE INDEX IF NOT EXISTS idx_indicators_dataset_variable
    ON county_indicators(dataset_id, variable);

-- Election data: one row per county-equivalent and presidential election year.
CREATE TABLE IF NOT EXISTS presidential_election_results (
    fips TEXT NOT NULL,
    year INTEGER NOT NULL,
    county_name TEXT,
    state_name TEXT,
    dem_votes INTEGER,
    rep_votes INTEGER,
    other_votes INTEGER,
    total_votes INTEGER,
    dem_pct REAL,
    rep_pct REAL,
    margin_pct REAL,
    source TEXT,
    PRIMARY KEY (fips, year)
);
CREATE INDEX IF NOT EXISTS idx_election_year ON presidential_election_results(year);

-- Retain source records and metadata without losing fields that change between releases.
CREATE TABLE IF NOT EXISTS source_records (
    record_id INTEGER PRIMARY KEY,
    dataset_id INTEGER NOT NULL REFERENCES dataset_catalog(dataset_id),
    fips TEXT,
    year INTEGER,
    record_json TEXT NOT NULL,
    imported_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_source_records_dataset
    ON source_records(dataset_id, fips, year);

CREATE TABLE IF NOT EXISTS data_load_log (
    load_id INTEGER PRIMARY KEY,
    dataset_name TEXT NOT NULL,
    source_file TEXT,
    rows_loaded INTEGER DEFAULT 0,
    status TEXT NOT NULL,
    message TEXT,
    loaded_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE VIEW IF NOT EXISTS v_county_election_2024 AS
SELECT fips, county_name, state_name, dem_votes, rep_votes, other_votes,
       total_votes, dem_pct, rep_pct, margin_pct, source
FROM presidential_election_results
WHERE year = 2024;

CREATE VIEW IF NOT EXISTS v_county_election_trends AS
SELECT fips, county_name, state_name, year, dem_pct, rep_pct, margin_pct,
       dem_votes, rep_votes, total_votes, source
FROM presidential_election_results;
"""

readme = """# USA Health Data Atlas: SQLite database

Local, persistent copy of the county data behind the
[USA Health Data Atlas](https://zia207.github.io/USA-Health-Data-Atlas/eda_model_development.html),
used by `Tutorials/notebooks/04_spatial_statistics/04_10_capstone/04_10_01_voting_pattern.ipynb`.

## Files

- `usa_health_atlas.sqlite`: the database (about 160 MB once populated).
- `build_database.py`: downloads the Atlas JavaScript bundles and parses each
  one with a dataset-specific loader (standard library only).
- `schema.sql`: database DDL.
- `atlas_assets/`: cache of the downloaded bundles (created on first build).

## Build

```bash
python Tutorials/notebooks/04_spatial_statistics/04_10_capstone/data_processing.py
# or, schema already present:
python Data/00_shared/data_spatial_stat/usa_health_atlas_database/build_database.py
```

## What is loaded

| Bundle | Table | Content |
|---|---|---|
| `presidential_election_data.js` | `presidential_election_results` | County returns 2004, 2008, 2012, 2016, 2020 (MEDSL) and 2024 (tonmcg, provisional) |
| `census_demography_data.js` | `county_indicators` | Census PEP 2010-2023: population, change, migration, age, sex, race/ethnicity, % rural |
| `svi_county_data.js` | `county_indicators` | CDC/ATSDR SVI 2022 percentile ranks (overall, themes, 16 EPL variables) |
| `religion_census_data.js` | `county_indicators` | Religion Census 2010 and 2020: adherents, congregations, rate per 1,000 for 10 dominant groups + total |
| `chr_county_sql_data.js` | `county_indicators` | County Health Rankings 2020-2025 releases: 29 measures + overall ranks/z-scores |

Each bundle's `meta` block (sources, notes, indicator definitions) is stored in
`source_records`; import status is in `data_load_log`.

## Example SQL

```sql
-- 2020 vs 2024 GOP two-party share by county
SELECT fips,
       MAX(CASE WHEN year = 2020 THEN 100.0 * rep_votes / (rep_votes + dem_votes) END) AS rep2p_2020,
       MAX(CASE WHEN year = 2024 THEN 100.0 * rep_votes / (rep_votes + dem_votes) END) AS rep2p_2024
FROM presidential_election_results
WHERE year IN (2020, 2024)
GROUP BY fips;

-- One indicator for one year
SELECT i.fips, i.value
FROM county_indicators i JOIN dataset_catalog d USING (dataset_id)
WHERE d.dataset_name = 'census_demography' AND i.variable = 'PCT_HISPANIC' AND i.year = 2020;
```

## Notes

1. FIPS is a five-character string (leading zeros kept).
2. 2024 returns are a compiled, non-certified feed; 2004-2020 are MEDSL. The
   `source` column keeps this distinction.
3. Connecticut 2024 results use planning-region FIPS that do not match its
   2020 county FIPS; Kansas City, MO is reported separately from Jackson County
   in the 2020 MEDSL data but not in the 2024 feed.
4. Units: CHR values are proportions (0-1); SVI values are percentile ranks
   (0-1); Religion Census rates are per 1,000 residents (small counties can
   exceed 1,000); Census PEP percentages are 0-100.
"""

datasets = [
("county_health_rankings", "County Health Rankings", "CHR", "county", "2020–2025", "Measures, ranks and outcomes; selected year in Atlas pipeline."),
("social_vulnerability_index", "CDC/ATSDR", "CDC/ATSDR SVI", "county", "2022", "County-level SVI percentile ranks."),
("health_experience_places", "CDC PLACES", "CDC", "county", "Current release", "Crude prevalence estimates for health outcomes and risk factors."),
("mortality", "NIMHD HDPulse", "NIMHD", "county", "2019–2023", "Age-adjusted mortality rates per 100,000."),
("census_demography", "U.S. Census Population Estimates Program", "U.S. Census Bureau", "county", "2010–2023", "Population, population change, births, deaths, migration, age, sex, race/ethnicity and rurality."),
("religion_census", "U.S. Religion Census / RCMS", "Association of Statisticians of American Religious Bodies", "county", "2010, 2020", "Adherents, congregations, rates and population shares for major traditions."),
("presidential_election", "MEDSL (2004–2020); tonmcg (2024)", "MIT Election Data & Science Lab / tonmcg", "county", "2004, 2008, 2012, 2016, 2020, 2024", "County presidential returns; 2024 source is compiled and documented as provisional."),
("cancer_incidence", "NCI State Cancer Profiles", "National Cancer Institute", "county", "2020–2025; pooled 2017–2021", "Synthetic annual incidence and observed pooled incidence."),
("cancer_mortality", "NCI State Cancer Profiles", "National Cancer Institute", "county", "2019–2023", "Reconstructed county cancer mortality."),
("overdose_mortality", "CDC/NCHS", "CDC", "county", "2003 onward, depending on layer", "Model-based, provisional, and opioid mortality layers."),
("soil_samples", "USGS", "USGS", "state / sample", "Varies", "Optional advanced soil sample table."),
]
datasets.append(("atlas_eda_pipeline", "https://zia207.github.io/USA-Health-Data-Atlas/eda_model_development.html",
                 "USA Health Data Atlas", "county", "Multiple",
                 "Database design derived from the EDA/model-development data pipeline."))


def init_database():
    """Create tables/views and the dataset catalog; write schema.sql and README.md."""
    con = sqlite3.connect(db_path)
    con.executescript(schema)
    con.executemany("""INSERT OR IGNORE INTO dataset_catalog
        (dataset_name, source_page, source_provider, geography_level, temporal_coverage, notes)
        VALUES (?, ?, ?, ?, ?, ?)""", datasets)
    con.commit()
    con.close()
    (out / "schema.sql").write_text(schema, encoding="utf-8")
    (out / "README.md").write_text(readme, encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description="Create and populate the USA Health Data Atlas SQLite database.")
    p.add_argument("--schema-only", action="store_true", help="create schema and catalog, skip the data import")
    p.add_argument("--assets-dir", default=str(out / "atlas_assets"), help="cache for downloaded bundles")
    args = p.parse_args()
    init_database()
    print("Schema and catalog ready:", db_path)
    if args.schema_only:
        return
    spec = importlib.util.spec_from_file_location("build_database", out / "build_database.py")
    build_database = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build_database)
    build_database.populate(db_path, Path(args.assets_dir))
    print(f"Database populated: {db_path} ({db_path.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
