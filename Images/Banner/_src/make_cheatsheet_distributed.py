"""Render the Distributed Spatial Computing quick reference (two columns, tutorials below) as one PNG.

Reuses the layout, CSS and logos of make_cheatsheet_spatial_analysis.py. Functions were chosen from
the calls the 08_distributed_spatial notebooks make; every name is checked against the package.

    python Images/Banner/_src/make_cheatsheet_distributed.py
"""

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
OUT = HERE.parent / "08_00_distributed_spatial_quick_reference.png"
COPY = ROOT / "Tutorials" / "notebooks" / "08_distributed_spatial" / "08_distributed_spatial_quick_reference.png"

# Each group: (heading, tag, rows). The tag is the module under geolattice.distributed (imported as pdist).
FOUNDATION = [
    ("Sessions and clusters", "pdist", [
        ("start · Session", "local, cuda, slurm, pbs, kubernetes or an address"),
        ("status · dashboard_link · get_session · shutdown", "Inspect and close the active cluster"),
        ("capabilities", "Which engines are installed, how many GPUs"),
    ]),
    ("Backends and laziness", "pdist", [
        ("set_defaults · get_defaults", "Default compute, chunks and partitions"),
        ("to_gpu · to_cpu · is_gpu", "Move arrays between host and CUDA"),
        ("persist · compute · is_lazy", "Keep results in memory or materialise them"),
    ]),
    ("Partitioning and dispatch", "pdist", [
        ("spatial_partition · partition_bounds", "Spatially compact vector partitions"),
        ("plan_chunks", "Square spatial chunks for rasters"),
        ("list_ops · dispatch · explain", "Which implementation runs, and why"),
    ]),
    ("Parallel I/O", "pdist", [
        ("read_raster · write_raster", "Lazy chunked rasters, Zarr or GeoTIFF out"),
        ("read_vector · write_vector", "Partitioned GeoParquet and OGR"),
        ("read_zarr · read_stac", "Zarr stores and STAC items as xarray"),
    ]),
    ("Raster", "pdist.raster", [
        ("calc · where · reclassify · normalized_difference", "Map algebra"),
        ("focal · convolve", "Moving windows with halos"),
        ("slope · aspect · hillshade · curvature", "Terrain on chunks or GPU"),
        ("tri · tpi · roughness", "Terrain indices"),
        ("map_tiles", "Your own function, chunk by chunk"),
    ]),
    ("Vector", "pdist.vector", [
        ("area · length · bounds · centroid", "Per-feature geometry, GPU kernels"),
        ("buffer · simplify · make_valid · to_crs", "Geometry operations"),
        ("clip · sjoin", "Clip and partition-pruned spatial join"),
        ("knn", "k nearest neighbours"),
    ]),
]

ANALYSIS = [
    ("Raster and vector together", "pdist", [
        ("rasterize · polygonize", "Features to cells and back"),
        ("zonal_stats · extract_values", "Zonal statistics, values at points"),
    ]),
    ("Spatial statistics and clustering", "pdist.analysis", [
        ("weights · morans_i · gearys_c", "Weights and global autocorrelation"),
        ("local_morans · getis_ord_local", "LISA and hot spots"),
        ("dbscan · hdbscan · kmeans", "Clustering large point sets"),
        ("kde", "Kernel density"),
    ]),
    ("Networks", "pdist.network", [
        ("graph_from_lines · nearest_nodes", "Graphs from road lines, snap points"),
        ("shortest_path · shortest_path_lengths", "Routes and distances"),
        ("centrality · components", "Centrality and connected components"),
    ]),
    ("Hydrology", "pdist.hydrology", [
        ("flow_direction", "Chunked D8 flow direction"),
        ("fill_depressions · flow_accumulation", "Depression filling, flow accumulation"),
    ]),
    ("Machine learning", "pdist.ml", [
        ("fit · predict_raster", "Fit a model, predict every pixel"),
        ("predict_tiles", "Tiled deep learning inference on GPUs"),
        ("train_ddp · ddp_model · gpu_count", "Multi-GPU training with DDP"),
    ]),
    ("Apache Sedona", "pdist.sedona", [
        ("spark · packages · stop", "Sedona-enabled Spark session"),
        ("to_spark · to_geopandas · is_spark", "Move data between Spark and GeoPandas"),
        ("sjoin", "Spatial join on Spark"),
    ]),
    ("Diagnostics", "pdist", [
        ("benchmark · check_equivalence", "Time CPU against GPU, compare results"),
        ("memory_report · profile", "Memory use and where the time goes"),
    ]),
]

NOTEBOOK_LIST = [
    ("08_00", "Why Data Outgrow One Core", "Fundamentals and the three rules"),
    ("08_01", "Sessions and Clusters", "Workers and the dashboard"),
    ("08_02", "Parallel Input and Output", "Big rasters, vectors, Zarr, STAC"),
    ("08_03", "Rasters at Scale", "Map algebra, focal, terrain on GPU"),
    ("08_04", "Vectors at Scale", "Geometry, clip, joins, neighbours"),
    ("08_05", "Rasters and Vectors Together", "Rasterize, zonal stats, polygonize"),
    ("08_06", "Spatial Statistics and Clustering", "Weights, Moran, hot spots"),
    ("08_07", "Networks", "Shortest paths, centrality, access"),
    ("08_08", "Hydrology", "Flow on large DEMs"),
    ("08_09", "Machine Learning", "Forests, boosting, deep learning"),
    ("08_10", "Apache Sedona", "Spatial SQL on Spark"),
    ("08_11", "Diagnostics and Performance", "Measure, check and tune"),
    ("08_12", "Capstone", "Soil carbon and cropland, Great Plains"),
]

IMPORTS = ('import geolattice.distributed as pdist<br>'
           'with pdist.start("cuda"):  # one worker per GPU<br>'
           '&nbsp;&nbsp;&nbsp;&nbsp;dem = pdist.read_raster("dem.tif"); s = pdist.raster.slope(dem, compute="gpu")')

tut = "".join(f'<div class="t"><b>{n}</b><div>{ti}<small>{d}</small></div></div>' for n, ti, d in NOTEBOOK_LIST)

HTML = f"""<!doctype html><html><head><meta charset="utf-8"><style>{sa.CSS}</style></head><body><div id="page">
<div class="bar"></div>
<div class="upatta"><img src="data:image/png;base64,{base.UPATTA}"></div>
<div class="kick"><div class="logo"><img src="data:image/png;base64,{base.LOGO}"></div>
<div class="k">QUICK REFERENCE · DR. ZIA U. AHMED</div></div>
<h1>Distributed Spatial <span>with GeoLattice</span></h1>
<div class="sub">Key functions of <b>geolattice.distributed</b>, grouped by task, with the tutorials that teach them.</div>
<div class="imps">{IMPORTS}</div>
<div class="cols">
{sa.column("PART A · ENGINE &amp; DATA", "Clusters, I/O, rasters &amp; vectors", FOUNDATION)}
{sa.column("PART B · ANALYSIS AT SCALE", "Statistics, networks, ML &amp; Spark", ANALYSIS)}
</div>
<div class="tutall"><h3>Tutorials · Tutorials/notebooks/08_distributed_spatial · {len(NOTEBOOK_LIST)} notebooks</h3><div class="tgrid">{tut}</div></div>
<div class="foot"><span>DISTRIBUTED SPATIAL COMPUTING WITH GEOLATTICE · UPATTA ANALYTICS</span>
<span>every op takes compute="auto" | "cpu" | "gpu" · pip install "geolattice[distributed]"</span></div>
</div></body></html>"""


def check_names():
    """Return the names on the card that geolattice.distributed does not provide."""
    import geolattice.distributed as pdist

    bad = []
    for groups in (FOUNDATION, ANALYSIS):
        for _, tag, rows in groups:
            mod = pdist if tag == "pdist" else getattr(pdist, tag.split(".", 1)[1])
            bad += [f"{tag}: {n}" for names, _ in rows for n in names.split(" · ") if not hasattr(mod, n)]
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
