"""Render a two column quick reference (vector | raster functions and tutorials) as one PNG.

    python Images/Banner/_src/make_cheatsheet_vector_raster.py
"""

import base64
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "01_02_vector_raster_quick_reference.png"
LOGO = base64.b64encode((HERE.parent.parent / "geolattice_logo.png").read_bytes()).decode()


def _trimmed_png(path, pad=6):
    """Logo cropped to its non-white content, as base64 PNG."""
    import io

    import numpy as np

    im = Image.open(path).convert("RGB")
    a = np.asarray(im).astype(int)
    ink = (255 * 3 - a.sum(2)) > 40  # pixels noticeably darker than white
    rows, cols = np.flatnonzero(ink.any(1)), np.flatnonzero(ink.any(0))
    box = (max(cols[0] - pad, 0), max(rows[0] - pad, 0), min(cols[-1] + pad, im.width), min(rows[-1] + pad, im.height))
    buf = io.BytesIO()
    im.crop(box).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


UPATTA = _trimmed_png(HERE.parent.parent / "upatta_logo.png")

VECTOR = {
    "Read &amp; write": [
        ("read_vector_data", "Shapefile, GeoPackage, GeoJSON, CSV, … via GDAL/OGR"),
        ("write_vector_data", "Write features to any supported driver"),
        ("write_geopackage · write_geojson", "Format specific writers"),
        ("supported_formats", "Formats and drivers available"),
    ],
    "Cloud native": [
        ("open_cloud_native_vector_data", "GeoParquet, FlatGeobuf, PMTiles, lakehouse tables"),
        ("read_geoparquet · write_geoparquet", "Columnar GeoParquet, optional bbox filter"),
        ("write_flatgeobuf", "FlatGeobuf with a packed Hilbert R-tree"),
        ("write_pmtiles", "PMTiles archive of vector tiles"),
    ],
    "Coordinate reference systems": [
        ("define_vector_crs", "CRS from EPSG, WKT, PROJ or pyproj"),
        ("reproject_vector", "Reproject features to a target CRS"),
        ("align_vector_crs", "Bring a second layer to the first layer's CRS"),
        ("crs_equal · is_geographic", "Compare and inspect CRS"),
    ],
    "Geoprocessing": [
        ("clip · intersect · union · erase", "Overlay two layers"),
        ("dissolve", "Merge features by attribute"),
        ("buffer · convex_hull", "Zones and hulls around features"),
    ],
    "Geometry tools": [
        ("centroids · point_on_surface", "Representative points"),
        ("points_along_geometry · transect", "Sample and cross lines"),
        ("fix_geometries · check_validity", "Find and repair invalid geometry"),
        ("rotate · translate · affine_transform", "Move and transform features"),
    ],
    "Attributes &amp; analysis": [
        ("filter_features · calculate_fields", "Select rows, compute fields"),
        ("join_attributes · statistics_by_categories", "Join tables, grouped statistics"),
        ("count_points_in_polygon · distance_matrix", "Point counts and distances"),
        ("nearest_neighbour_analysis", "Clark Evans clustering test"),
        ("kmeans_clustering · dbscan_clustering", "Cluster point features"),
    ],
    "Maps (geolattice.viz)": [
        ("plot_layer · plot_pair", "One layer, or two side by side"),
        ("apply_map_style", "Tutorial map theme"),
    ],
}

RASTER = {
    "Read &amp; write": [
        ("open_raster_data · read_raster_data", "Any GDAL raster as a RasterDataset"),
        ("read_raster", "GeoTIFF, COG or URL as an xarray DataArray"),
        ("write_raster_data · write_geotiff", "Write with any GDAL raster driver"),
        ("write_png · write_jpeg", "Quick looks and figures"),
    ],
    "CRS &amp; warping": [
        ("get_raster_crs · assign_raster_crs", "Read or set the CRS without warping"),
        ("reproject_raster · warp", "Reproject and resample with GDAL Warp"),
        ("list_datum_transformations", "Datum transformation paths"),
    ],
    "Create &amp; edit": [
        ("create · edit · convert", "New rasters, metadata, format conversion"),
        ("resize · unscale", "Change size, apply scale and offset"),
        ("fill_nodata", "Interpolate nodata gaps from edges"),
    ],
    "Algebra &amp; neighbourhoods": [
        ("calc · raster_algebra", "Band math expressions"),
        ("raster_ndvi", "(NIR − red) / (NIR + red)"),
        ("focal_statistics", "Moving window statistics"),
        ("blend · compare", "Compose or difference two rasters"),
    ],
    "Classify &amp; summarise": [
        ("reclassify · reclassify_by", "Recode values, or n classes by scheme"),
        ("class_intervals · raster_histogram", "Breaks and histograms"),
        ("summarize_raster · zonal_stats", "Global and zonal statistics"),
    ],
    "Resample &amp; combine": [
        ("resample · aggregate · disaggregate", "Change resolution"),
        ("clip · clip_to_bbox", "Cut by bounds, window or polygon"),
        ("mosaic · stack", "Join tiles, combine bands"),
    ],
    "Raster ⇄ vector": [
        ("rasterize · polygonize", "Burn features into cells, cells into polygons"),
        ("proximity · contour", "Distance maps, contour lines"),
        ("color_map", "Colour relief from a single band"),
    ],
    "Maps (geolattice.viz)": [
        ("plot_raster · plot_rasters", "One raster, or several panels"),
        ("plot_raster_histogram", "Band value histograms"),
    ],
}

TUT_V = [
    ("01_00", "Vector Data", "Features, geometry types, attributes, formats"),
    ("01_01", "Reading and Writing Vector Data", "Files, databases and cloud native formats"),
    ("01_02", "Coordinate Reference Systems", "Define, compare and reproject"),
    ("01_03", "Visualizing Vector Data", "Static and thematic maps"),
    ("01_04", "Geoprocessing", "Clip, overlay, dissolve, buffer"),
    ("01_05", "Geometry Operations", "Centroids, repair, transforms, analysis"),
]
TUT_R = [
    ("02_00", "Raster Data", "Cells, bands, resolution, nodata, formats"),
    ("02_01", "Reading and Writing Raster Data", "GDAL datasets, GeoTIFF, COG, xarray"),
    ("02_02", "Coordinate Reference Systems", "Assign, warp and reproject rasters"),
    ("02_03", "Raster Operations", "Algebra, classify, resample, zonal stats"),
]

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;700;800&family=JetBrains+Mono:wght@500;700&display=swap');
* { margin: 0; padding: 0; box-sizing: border-box; }
html, body { width: 1000px; }
body { font-family: 'Plus Jakarta Sans', 'Noto Sans', 'DejaVu Sans', sans-serif; color: #fff;
  background:
    linear-gradient(rgba(255,255,255,0.04) 1px, transparent 1px) 0 0 / 22px 22px,
    linear-gradient(90deg, rgba(255,255,255,0.04) 1px, transparent 1px) 0 0 / 22px 22px,
    linear-gradient(180deg, #1f4290 0%, #17346f 55%, #122a5c 100%); }
#page { position: relative; padding: 30px 36px 26px 44px; }
.bar { position: absolute; left: 0; top: 0; bottom: 0; width: 7px; background: #f5c518; }
.upatta { position: absolute; top: 22px; right: 36px; background: #fff; border-radius: 10px; padding: 6px 8px;
  box-shadow: 0 2px 10px rgba(0,0,0,0.25); }
.upatta img { display: block; height: 82px; }
.kick { display: flex; align-items: center; gap: 12px; }
.logo { width: 34px; height: 34px; border-radius: 8px; background: #fff; padding: 2px; }
.logo img { width: 30px; height: 30px; border-radius: 50%; }
.k { font-size: 10.5px; font-weight: 800; letter-spacing: 2.6px; color: #f5c518; }
h1 { margin-top: 12px; font-size: 32px; font-weight: 800; letter-spacing: -0.5px; line-height: 1.05; }
h1 span { color: #f5c518; }
.sub { margin-top: 7px; font-size: 12.5px; color: #dbe5ff; }
.cols { margin-top: 20px; display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
.col { background: rgba(10,28,72,0.42); border: 1px solid rgba(143,179,255,0.28); border-radius: 14px; padding: 16px 16px 14px; }
.colhead { display: flex; align-items: baseline; justify-content: space-between; margin-bottom: 10px; }
.colhead h2 { font-size: 19px; font-weight: 800; }
.colhead h2 em { font-style: normal; color: #f5c518; font-size: 10.5px; letter-spacing: 2px; display: block; margin-bottom: 2px; }
code.imp { font-family: 'JetBrains Mono', 'DejaVu Sans Mono', monospace; font-size: 10px; color: #13306a;
  background: #f5c518; padding: 3px 8px; border-radius: 6px; font-weight: 700; }
.grp { margin-top: 10px; }
.grp h3 { font-size: 10px; font-weight: 800; letter-spacing: 1.8px; text-transform: uppercase; color: #8fb3ff; margin-bottom: 5px; }
.fn { display: grid; grid-template-columns: 52% 48%; gap: 8px; padding: 3px 0; border-bottom: 1px dashed rgba(143,179,255,0.16); }
.fn:last-child { border-bottom: none; }
.fn code { font-family: 'JetBrains Mono', 'DejaVu Sans Mono', monospace; font-size: 9.6px; color: #ffd84d; font-weight: 700; }
.fn span { font-size: 10px; color: #dbe5ff; line-height: 1.3; }
.tut { margin-top: 14px; padding-top: 10px; border-top: 1px solid rgba(143,179,255,0.3); }
.tut h3 { font-size: 10px; font-weight: 800; letter-spacing: 1.8px; text-transform: uppercase; color: #f5c518; margin-bottom: 6px; }
.t { display: grid; grid-template-columns: 44px 1fr; gap: 6px; align-items: start; padding: 3px 0; }
.t b { font-family: 'JetBrains Mono', monospace; font-size: 9.5px; background: rgba(255,255,255,0.12);
  border: 1px solid rgba(255,255,255,0.25); border-radius: 999px; text-align: center; padding: 2px 0; }
.t div { font-size: 11px; font-weight: 700; }
.t div small { display: block; font-weight: 500; font-size: 9.8px; color: #c9d8ff; }
.foot { margin-top: 14px; display: flex; justify-content: space-between; font-size: 8.5px; font-weight: 700;
  letter-spacing: 1.4px; color: rgba(219,229,255,0.55); }
"""


def column(part, title, imp, groups, tuts, folder):
    g = "".join(f'<div class="grp"><h3>{h}</h3>' + "".join(
        f'<div class="fn"><code>{n}</code><span>{d}</span></div>' for n, d in items) + "</div>"
        for h, items in groups.items())
    t = "".join(f'<div class="t"><b>{n}</b><div>{ti}<small>{d}</small></div></div>' for n, ti, d in tuts)
    return (f'<div class="col"><div class="colhead"><h2><em>{part}</em>{title}</h2><code class="imp">{imp}</code></div>'
            f'{g}<div class="tut"><h3>Tutorials · {folder}</h3>{t}</div></div>')


HTML = f"""<!doctype html><html><head><meta charset="utf-8"><style>{CSS}</style></head><body><div id="page">
<div class="bar"></div>
<div class="upatta"><img src="data:image/png;base64,{UPATTA}"></div>
<div class="kick"><div class="logo"><img src="data:image/png;base64,{LOGO}"></div>
<div class="k">QUICK REFERENCE · DR. ZIA U. AHMED</div></div>
<h1>Vector &amp; Raster Data <span>with GeoLattice</span></h1>
<div class="sub">Key functions of <b>geolattice.vector</b> and <b>geolattice.raster</b>, grouped by task, with the tutorials that teach them.</div>
<div class="cols">
{column("PART 1 · FEATURES", "Vector data", "import geolattice.vector as pv", VECTOR, TUT_V, "Tutorials/notebooks/01_vector")}
{column("PART 2 · GRIDS", "Raster data", "import geolattice.raster as pr", RASTER, TUT_R, "Tutorials/notebooks/02_raster")}
</div>
<div class="foot"><span>VECTOR &amp; RASTER DATA WITH GEOLATTICE · UPATTA ANALYTICS</span>
<span>call any function as pv.name(...) or pr.name(...)</span></div>
</div></body></html>"""

if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "sheet.html"
        src.write_text(HTML)
        raw = Path(tmp) / "raw.png"
        subprocess.run(["google-chrome", "--headless=new", "--disable-gpu", "--hide-scrollbars",
                        "--force-device-scale-factor=2", "--window-size=1000,1600",
                        "--virtual-time-budget=4000", f"--screenshot={raw}", src.as_uri()],
                       check=True, capture_output=True)
        im = Image.open(raw).convert("RGB")
        # crop the empty page below the content: last row that differs from the bottom background row
        import numpy as np

        a = np.asarray(im).astype(int)
        bar = a[:, 6]  # the yellow side bar spans exactly the page
        yellow = (bar[:, 0] > 200) & (bar[:, 1] > 170) & (bar[:, 2] < 80)
        last = int(np.flatnonzero(yellow).max()) + 1
        im.crop((0, 0, im.width, last)).save(OUT)
    print("wrote", OUT, Image.open(OUT).size)
