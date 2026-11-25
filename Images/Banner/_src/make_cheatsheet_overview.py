"""Render the GeoLattice overview quick reference (every module on one card) as one PNG.

Reuses the layout, CSS and logos of make_cheatsheet_spatial_analysis.py. Each group shows a few key
names from one module; every name is checked against the installed package. Notebook counts are
counted from Tutorials/notebooks at render time.

    python Images/Banner/_src/make_cheatsheet_overview.py
"""

import importlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_cheatsheet_spatial_analysis as sa  # noqa: E402

base = sa.base
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TUTORIALS = ROOT / "Tutorials" / "notebooks"
OUT = HERE.parent / "00_geolattice_quick_reference.png"
COPY = TUTORIALS / "00_geolattice_quick_reference.png"

# Tag -> module it refers to. A group tag like "sna · seg" means each name carries its alias prefix.
MODULES = {
    "crs": "geolattice.crs",
    "pv": "geolattice.vector",
    "pr": "geolattice.raster",
    "viz": "geolattice.viz",
    "sa": "geolattice.spatial_analysis",
    "sna": "geolattice.spatial_analysis.network_analysis",
    "seg": "geolattice.spatial_analysis.segregation",
    "sc": "geolattice.spatial_analysis.spatialcausal",
    "sstat": "geolattice.spatial_statistics",
    "gstat": "geolattice.geostatistics",
    "pdb": "geolattice.geolatticedb",
    "web": "geolattice.web",
    "pdist": "geolattice.distributed",
    "rs": "geolattice.rsense",
    "ic": "geolattice.rsense.classify.image_classification",
    "fm": "geolattice.rsense.foundation_models",
    "seb": "geolattice.rsense.surface_energy_balance",
}

ANALYSIS = [
    ("Coordinate reference systems", "crs", [
        ("CRS.from_epsg · CRS.from_wkt · CRS.to_epsg", "Build and inspect a CRS"),
        ("CustomProjection · DatumTransformation", "Custom projections, datum paths"),
        ("query_crs_info", "Search EPSG and ESRI codes"),
    ]),
    ("Vector data", "pv", [
        ("read_vector_data · write_vector_data · write_geoparquet", "Desktop and cloud-native formats"),
        ("reproject_vector · define_vector_crs", "Assign and reproject CRS"),
        ("clip · buffer · dissolve · intersect", "Geoprocessing"),
        ("centroids · fix_geometries · count_points_in_polygon", "Geometry and attribute tools"),
    ]),
    ("Raster data", "pr", [
        ("open_raster · read_raster_data · write_geotiff", "GDAL raster I/O"),
        ("reproject_raster · warp", "Reproject and resample"),
        ("calc · reclassify · resample · mosaic", "Algebra and combining"),
        ("zonal_stats · rasterize · polygonize", "Raster and vector together"),
    ]),
    ("Maps", "viz", [
        ("plot_layer · plot_raster · plot_rgb", "Vector, raster and colour composite maps"),
        ("plot_choropleth · interactive_map", "Thematic and interactive maps"),
        ("apply_map_style · add_scale_bar · add_north_arrow", "Map style and elements"),
    ]),
    ("Spatial analysis", "sa", [
        ("slope · hillshade · twi", "Terrain"),
        ("fill_depressions · d8_pointer · watershed", "Surface hydrology"),
        ("weighted_overlay · fuzzy_overlay", "Multi-criteria suitability"),
        ("MultidimensionalRaster · generate_trend_raster", "Data cubes and trends"),
    ]),
    ("Networks and segregation", "sna · seg", [
        ("sna.load_road_graph · sna.shortest_routes · sna.isochrone_polygons", "Routing and service areas"),
        ("seg.compute · seg.local_significance", "Segregation indices, local tests"),
    ]),
    ("Spatial causal inference", "sc", [
        ("SpatialDML · SpatialPlus", "Spatial confounding"),
        ("InterferenceEstimator · GWDML", "Spillovers, varying effects"),
        ("GeographicRDD · SpatialDiD · SpatialSyntheticControl", "Quasi-experiments"),
    ]),
]

PLATFORMS = [
    ("Spatial statistics", "sstat", [
        ("weights.Queen · weights.KNN", "Spatial weights"),
        ("esda.Moran · esda.Moran_Local", "Global and local autocorrelation"),
        ("regression.SpatialLag · regression.SpatialError", "Spatial regression"),
        ("gwmodel.GWR · gwmodel.MGWR", "Geographically weighted models"),
        ("pointpattern.ripley_k · cluster.getis_ord_gi", "Point patterns and hot spots"),
        ("bayes.BYM · interpolate.IDW", "Disease mapping, interpolation"),
    ]),
    ("Geostatistics", "gstat", [
        ("Variogram · OrdinaryKriging · UniversalKriging", "Variograms and kriging"),
        ("RegressionKriging · IndicatorKriging", "Covariates and thresholds"),
        ("PoissonKriging · EmpiricalBayesianKriging", "Areal rates, Bayesian kriging"),
        ("sgsim · sisim", "Geostatistical simulation"),
    ]),
    ("Spatial databases", "pdb", [
        ("connect · BACKENDS", "PostGIS, SpatiaLite, GeoPackage, DuckDB"),
        ("load · query · spatial_join · nearest", "Load, query and join in the database"),
    ]),
    ("Web and cloud GIS", "web", [
        ("to_cog · validate_cog", "Cloud-Optimized GeoTIFF"),
        ("write_geoparquet · write_pmtiles", "Cloud-native vectors"),
        ("make_xyz_tiles · create_app · start_server", "Tiles, OGC API Features, viewer"),
    ]),
    ("Distributed computing", "pdist", [
        ("start · capabilities", "Local, GPU, HPC or Kubernetes clusters"),
        ("read_raster · raster.slope · vector.sjoin", "Chunked rasters, partitioned vectors"),
        ("zonal_stats · ml.predict_raster · sedona.sjoin", "Analysis, ML and Spark at scale"),
    ]),
    ("Remote sensing", "rs", [
        ("read · set_backend", "Read imagery, choose CPU or GPU"),
        ("optical.load_sentinel2 · optical.index", "Optical scenes and indices"),
        ("sar.to_db · insar.unwrap", "Radar and interferometry"),
        ("lidar.chm · hyperspectral.mnf", "LiDAR and hyperspectral"),
    ]),
    ("Deep learning and ET", "ic · fm · seb", [
        ("ic.train_model · ic.predict_map", "Image classification"),
        ("fm.load_model · fm.linear_probe", "Foundation model embeddings"),
        ("seb.run_metric · seb.run_ssebop", "Evapotranspiration"),
    ]),
]

# (folder, badge, title, summary)
CHAPTERS = [
    ("01_02_vector_raster", "01-02", "Vector &amp; Raster Data", "Read, write, project, map and process"),
    ("03_spatial_analysis", "03", "Spatial Analysis", "Terrain, cubes, suitability, networks, causal"),
    ("04_spatial_statistics", "04", "Spatial Statistics", "Autocorrelation, regression, GWR, sampling"),
    ("05_geostatistics", "05", "Geostatistics", "Variograms, kriging, simulation, Bayesian"),
    ("06_spatialDB", "06", "Spatial Databases", "PostGIS, SpatiaLite, DuckDB, pgRouting"),
    ("07_web_cloud_gis", "07", "Web &amp; Cloud GIS", "COG, GeoParquet, tiles, APIs, apps"),
    ("08_distributed_spatial", "08", "Distributed Computing", "Dask clusters, GPUs, Sedona"),
    ("09_remote_sensing", "09", "Remote Sensing", "Optical, SAR, LiDAR, deep learning, ET"),
]

IMPORTS = ("import geolattice.vector as pv, geolattice.raster as pr, geolattice.spatial_analysis as sa<br>"
           "from geolattice.spatial_analysis import network_analysis as sna, segregation as seg, spatialcausal as sc<br>"
           "import geolattice.spatial_statistics as sstat, geolattice.geostatistics as gstat, geolattice.web as web<br>"
           "import geolattice.distributed as pdist, geolattice.rsense as rs; from geolattice import geolatticedb as pdb<br>"
           "import geolattice.crs as crs, geolattice.viz as viz, geolattice.rsense.classify.image_classification as ic<br>"
           "from geolattice.rsense import foundation_models as fm, surface_energy_balance as seb")


def count_notebooks(folder):
    # skip checkpoints and _-prefixed scratch notebooks (e.g. _wv_test.ipynb), which are not lessons
    return sum(1 for p in (TUTORIALS / folder).rglob("*.ipynb")
               if ".ipynb_checkpoints" not in p.parts and not p.name.startswith("_"))


counts = {folder: count_notebooks(folder) for folder, *_ in CHAPTERS}
total = sum(counts.values())

CSS = sa.CSS + """
.t { grid-template-columns: 70px 1fr; }
"""

tut = "".join(f'<div class="t"><b>{badge}</b><div>{title}<small><em>{counts[folder]} notebooks</em> · {d}</small></div></div>'
              for folder, badge, title, d in CHAPTERS)

HTML = f"""<!doctype html><html><head><meta charset="utf-8"><style>{CSS}</style></head><body><div id="page">
<div class="bar"></div>
<div class="upatta"><img src="data:image/png;base64,{base.UPATTA}"></div>
<div class="kick"><div class="logo"><img src="data:image/png;base64,{base.LOGO}"></div>
<div class="k">QUICK REFERENCE · DR. ZIA U. AHMED</div></div>
<h1>GeoLattice <span>at a glance</span></h1>
<div class="sub">One library for geospatial analysis in Python: a few key names from every module, with the chapters that teach them.</div>
<div class="imps">{IMPORTS}</div>
<div class="cols">
{sa.column("PART A · DATA &amp; ANALYSIS", "CRS, vector, raster, maps &amp; analysis", ANALYSIS)}
{sa.column("PART B · STATISTICS &amp; PLATFORMS", "Statistics, databases, cloud &amp; imagery", PLATFORMS)}
</div>
<div class="tutall"><h3>Tutorials · Tutorials/notebooks · {total} notebooks in {len(CHAPTERS)} chapters · each chapter has its own quick reference card</h3><div class="tgrid">{tut}</div></div>
<div class="foot"><span>GEOLATTICE · UPATTA ANALYTICS</span>
<span>pip install "geolattice[all]" · tags show the alias each name is called with</span></div>
</div></body></html>"""


def check_names():
    """Return the names on the card that do not resolve in the installed package."""
    bad = []
    for groups in (ANALYSIS, PLATFORMS):
        for _, tag, rows in groups:
            for names, _ in rows:
                for name in names.split(" · "):
                    alias, rest = (name.split(".", 1) if " · " in tag else (tag, name))
                    *subs, attr = rest.split(".")
                    obj = importlib.import_module(MODULES[alias])
                    try:
                        for part in subs:
                            obj = getattr(obj, part) if hasattr(obj, part) else importlib.import_module(
                                f"{obj.__name__}.{part}")
                        if not hasattr(obj, attr):
                            bad.append(f"{tag}: {name}")
                    except Exception:
                        bad.append(f"{tag}: {name}")
    return bad


if __name__ == "__main__":
    missing = check_names()
    if missing:
        raise SystemExit(f"names not found: {missing}")
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "sheet.html"
        src.write_text(HTML)
        raw = Path(tmp) / "raw.png"
        subprocess.run(["google-chrome", "--headless=new", "--disable-gpu", "--hide-scrollbars",
                        "--force-device-scale-factor=2", "--window-size=1400,3400",
                        "--virtual-time-budget=4000", f"--screenshot={raw}", src.as_uri()],
                       check=True, capture_output=True)
        im = Image.open(raw).convert("RGB")
        # crop below the content: the yellow side bar spans exactly the page
        bar = np.asarray(im).astype(int)[:, 6]
        yellow = (bar[:, 0] > 200) & (bar[:, 1] > 170) & (bar[:, 2] < 80)
        last = int(np.flatnonzero(yellow).max()) + 1
        if last >= im.height - 2:
            print("warning: page may be taller than the window; raise --window-size height")
        im.crop((0, 0, im.width, last)).save(OUT)
    shutil.copyfile(OUT, COPY)
    print("wrote", OUT, Image.open(OUT).size, "and", COPY, "| notebooks:", counts, total)
