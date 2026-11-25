"""Render the Web & Cloud GIS quick reference (geolattice.web | web stack, tutorials below) as one PNG.

Reuses the layout, CSS and logos of make_cheatsheet_spatial_analysis.py. geolattice names are checked
against the installed package; third-party names are checked against the code of the 07_web_cloud_gis
notebooks, their apps/ and the haor_monitor capstone package.

    python Images/Banner/_src/make_cheatsheet_web_gis.py
"""

import json
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
NOTEBOOKS = ROOT / "Tutorials" / "notebooks" / "07_web_cloud_gis"
OUT = HERE.parent / "07_00_web_cloud_gis_quick_reference.png"
COPY = NOTEBOOKS / "07_web_cloud_gis_quick_reference.png"

# geolattice tags -> module, checked with hasattr
MODULES = {"web": "geolattice.web", "rs.get_images": "geolattice.rsense.get_images"}

GEOLATTICE = [
    ("Cloud access", "web", [
        ("to_vsi_path · cloud_env · cloud_options", "Read S3, GCS, Azure or HTTP through GDAL"),
        ("remote_info", "Inspect a remote file without downloading it"),
    ]),
    ("Cloud-Optimized GeoTIFF", "web", [
        ("to_cog · validate_cog · COGReport", "Write a COG and check its layout"),
    ]),
    ("Cloud-native vectors", "web", [
        ("write_geoparquet · read_geoparquet · geoparquet_info", "GeoParquet with bbox covering"),
        ("write_flatgeobuf · write_pmtiles", "FlatGeobuf and PMTiles for the web"),
    ]),
    ("Web map tiles", "web", [
        ("lonlat_to_tile · tile_bounds · tiles_for_bounds", "XYZ tile maths"),
        ("tile_resolution · zoom_for_resolution", "Match zoom levels to pixel size"),
        ("render_tile · make_xyz_tiles", "Render one tile or a whole pyramid"),
        ("tilejson · raster_web_info · suggest_rescale", "Tile metadata and colour stretch"),
    ]),
    ("Serving", "web", [
        ("create_app", "XYZ tiles, OGC API Features and a Leaflet viewer"),
        ("serve · start_server", "Run with uvicorn, or in the background from a notebook"),
        ("leaflet_html", "Standalone Leaflet page"),
        ("static_server", "Local file server with range requests and CORS"),
    ]),
    ("Imagery search", "rs.get_images", [
        ("list_collections · search_images · bands_for", "Find scenes in public catalogues"),
        ("get_images", "Download matching scenes"),
    ]),
]

STACK = [
    ("STAC and cloud platforms", "pystac · openeo · ee", [
        ("pystac_client.Client.open · search", "Search a STAC API"),
        ("planetary_computer.sign_inplace · odc.stac.load", "Sign assets, load them as xarray"),
        ("pystac.Catalog · Collection", "Build and validate a static catalogue"),
        ("openeo.connect · load_collection", "Copernicus openEO back ends"),
        ("ee.Initialize · ImageCollection", "Google Earth Engine"),
    ]),
    ("Arrays at scale", "dask · zarr · rioxarray", [
        ("zarr.create_array · open_array", "Chunked arrays in object storage"),
        ("rio.write_crs · rio.clip_box · rio.to_raster", "rioxarray on xarray objects"),
        ("dask.array.from_array · LocalCluster", "Parallel and out-of-core processing"),
    ]),
    ("Spatial SQL over files", "duckdb", [
        ("duckdb.connect · sql · register", "Query GeoParquet and DataFrames in place"),
    ]),
    ("Interactive web maps", "folium · leafmap · lonboard", [
        ("folium.Map · TileLayer · GeoJson · LayerControl", "Leaflet maps from Python"),
        ("GeoJsonTooltip · GeoJsonPopup · CircleMarker", "Popups, tooltips and markers"),
        ("leafmap.Map · add_gdf · split_map · add_basemap", "Quick maps, swipe comparisons"),
        ("LonboardMap · ScatterplotLayer · PolygonLayer", "GPU maps for millions of features"),
    ]),
    ("OGC and tile clients", "owslib · pmtiles · httpx", [
        ("Features", "OGC API Features client (OWSLib)"),
        ("pmtiles.reader", "Read tiles from a PMTiles archive"),
        ("httpx.get · httpx.Client", "Call your own APIs and tile servers"),
    ]),
    ("Geospatial APIs", "fastapi", [
        ("FastAPI · app.get · app.post · HTTPException · Query", "Endpoints and validation"),
        ("BaseModel · Field · model_validator", "Request and response schemas (pydantic)"),
        ("Polygon · MultiPolygon", "GeoJSON types (geojson_pydantic)"),
        ("TestClient", "Test the app without a server"),
    ]),
    ("Web apps", "streamlit · panel · solara", [
        ("st.set_page_config · st.selectbox · st.plotly_chart", "Streamlit dashboards"),
        ("pn.extension · pn.Column · pn.Row", "Panel layouts"),
        ("solara.render", "Solara reactive apps"),
    ]),
    ("DevOps", "tools", [
        ("pyproject.toml · ruff · pytest", "Package, lint and test the pipeline"),
        ("Dockerfile · .github/workflows · cron", "Containers, CI and schedules"),
    ]),
]

NOTEBOOK_LIST = [
    ("07_00", "Web/Cloud GIS", "Concepts and architecture"),
    ("07_01", "Cloud-Native Rasters", "COG, Zarr and STAC"),
    ("07_02", "Cloud-Native Vectors", "GeoParquet, FlatGeobuf and PMTiles"),
    ("07_03", "Spatial SQL", "DuckDB, GeoPackage and PostGIS"),
    ("07_04", "Interactive Web Maps", "Folium, leafmap and lonboard"),
    ("07_05", "Serving Maps", "Tiles and OGC APIs"),
    ("07_06", "Geospatial APIs", "Building APIs with FastAPI"),
    ("07_07", "Web Apps", "Streamlit, Panel and Solara"),
    ("07_08", "Scalable Processing", "Dask, GPUs and batch jobs"),
    ("07_09", "Cloud Platforms", "Planetary Computer, Earth Engine, openEO"),
    ("07_10", "DevOps for Pipelines", "Packaging, testing, CI, containers, schedules"),
    ("07_11", "Capstone", "Cloud web GIS for flood monitoring in the Haor basin"),
]

IMPORTS = ('import geolattice.web as web<br>'
           'web.to_cog("ndvi.tif", "ndvi_cog.tif"); web.write_geoparquet("counties.parquet", counties)<br>'
           'app = web.create_app(rasters={"ndvi": "ndvi_cog.tif"}, vectors={"counties": "counties.parquet"})')

tut = "".join(f'<div class="t"><b>{n}</b><div>{ti}<small>{d}</small></div></div>' for n, ti, d in NOTEBOOK_LIST)

HTML = f"""<!doctype html><html><head><meta charset="utf-8"><style>{sa.CSS}</style></head><body><div id="page">
<div class="bar"></div>
<div class="upatta"><img src="data:image/png;base64,{base.UPATTA}"></div>
<div class="kick"><div class="logo"><img src="data:image/png;base64,{base.LOGO}"></div>
<div class="k">QUICK REFERENCE · DR. ZIA U. AHMED</div></div>
<h1>Web &amp; Cloud GIS <span>with GeoLattice</span></h1>
<div class="sub">The <b>geolattice.web</b> API and the web stack the tutorials use, with the notebooks that teach them.</div>
<div class="imps">{IMPORTS}</div>
<div class="cols">
{sa.column("PART A · GEOLATTICE", "Cloud data, tiles &amp; serving", GEOLATTICE)}
{sa.column("PART B · THE WEB STACK", "Catalogues, maps, APIs &amp; apps", STACK)}
</div>
<div class="tutall"><h3>Tutorials · Tutorials/notebooks/07_web_cloud_gis · {len(NOTEBOOK_LIST)} notebooks</h3><div class="tgrid">{tut}</div></div>
<div class="foot"><span>WEB &amp; CLOUD GIS WITH GEOLATTICE · UPATTA ANALYTICS</span>
<span>pip install "geolattice[web]" · part B tags name the library to import</span></div>
</div></body></html>"""


def check_names():
    """geolattice names must exist; third-party names must appear in the chapter's code or text."""
    import importlib

    text = ""
    for nb in NOTEBOOKS.glob("*.ipynb"):
        text += "\n".join("".join(c["source"]) for c in json.loads(nb.read_text())["cells"])
    for py in list(NOTEBOOKS.glob("apps/*.py")) + list(NOTEBOOKS.glob("haor_monitor/**/*.py")):
        text += py.read_text()
    bad = []
    for _, tag, rows in GEOLATTICE:
        mod = importlib.import_module(MODULES[tag])
        bad += [f"{tag}: {n}" for names, _ in rows for n in names.split(" · ") if not hasattr(mod, n)]
    for _, tag, rows in STACK:
        for names, _ in rows:
            for n in names.split(" · "):
                key = n if tag == "tools" else n.split(".")[-1]
                if key not in text:
                    bad.append(f"{tag}: {n}")
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
    print("wrote", OUT, Image.open(OUT).size, "and", COPY)
