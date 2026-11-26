"""Build the GeoLattice home page and publish it to docs/.

Run from anywhere: ``python Tutorials/home/build_home.py``.

1. Counts the notebooks each chapter site renders (from its _quarto.yml) and writes the
   chapter tiles and the totals to _generated/.
2. Writes getting_started.qmd from README.md, and user_guide.qmd and api_reference.qmd from
   docs/, with repository-relative links pointed at GitHub.
3. Renders the site with Quarto into _site/ and copies it to docs/, next to docs/notebooks/.
"""

import html
import posixpath
import re
import shutil
import subprocess
from pathlib import Path

import yaml

HOME = Path(__file__).resolve().parent  # Tutorials/home
REPO = HOME.parents[1]  # repository root
NOTEBOOKS = REPO / "Tutorials" / "notebooks"  # chapter Quarto projects
DOCS = REPO / "docs"  # GitHub Pages root
GEN = HOME / "_generated"  # generated includes
BLOB = "https://github.com/zia207/GeoLattice/blob/main/"  # links to repository files
RAW = "https://raw.githubusercontent.com/zia207/GeoLattice/main/"  # images from the repository

# (name, chapter folder, render prefix or None, anchor, module, description)
CHAPTERS = [
    ("Vector Data", "01_02_vector_raster", "01_vector/", "#vector-data", "geolattice.vector",
     "Read, write and reproject vector data, geoprocessing, geometry tools and attribute tables."),
    ("Raster Data", "01_02_vector_raster", "02_raster/", "#raster-data", "geolattice.raster",
     "Raster I/O and reprojection, map algebra, clip, mosaic, reclassify, zonal statistics and contours."),
    ("Spatial Analysis", "03_spatial_analysis", None, "", "geolattice.spatial_analysis",
     "Terrain, hydrology, data cubes, multi-criteria suitability, networks, segregation and spatial causal inference."),
    ("Spatial Statistics", "04_spatial_statistics", None, "", "geolattice.spatial_statistics",
     "Point patterns, autocorrelation, spatial regression, GWR, disease mapping, sampling and space-time analysis."),
    ("Geostatistics", "05_geostatistics", None, "", "geolattice.geostatistics",
     "Variograms, the kriging family, Poisson and Bayesian kriging, and geostatistical simulation."),
    ("Spatial Database", "06_spatialDB", None, "", "geolattice.geolatticedb",
     "PostGIS, SpatiaLite, GeoPackage and DuckDB: spatial SQL, joins, indexing, rasters and routing."),
    ("Web & Cloud GIS", "07_web_cloud_gis", None, "", "geolattice.web",
     "Cloud-native formats, web maps, tile and OGC API servers, web apps and scalable processing."),
    ("Distributed Spatial Analysis", "08_distributed_spatial", None, "", "geolattice.distributed",
     "Chunked rasters and partitioned vectors on all cores, Dask clusters, GPUs and Apache Sedona."),
    ("Remote Sensing", "09_remote_sensing", None, "", "geolattice.rsense",
     "Optical, radar, UAV, hyperspectral and LiDAR data, deep learning, foundation models, ET and soil moisture."),
]


def notebook_counts():
    """Notebooks each chapter renders, split by prefix for the shared vector and raster site."""
    counts = []  # one count per tile
    for _, folder, prefix, *_ in CHAPTERS:  # each tile
        render = yaml.safe_load((NOTEBOOKS / folder / "_quarto.yml").read_text())["project"]["render"]  # render list
        nbs = [r for r in render if r.endswith(".ipynb")]  # notebooks only
        counts.append(sum(r.startswith(prefix) for r in nbs) if prefix else len(nbs))  # count
    return counts


def write_generated(counts):
    """Chapter tiles and the totals line, included by index.qmd."""
    GEN.mkdir(exist_ok=True)  # output folder
    tiles = []  # one anchor per chapter
    for i, ((name, folder, _, anchor, module, desc), n) in enumerate(zip(CHAPTERS, counts), 1):  # each chapter
        tiles.append(
            f'<a class="gl-tile" href="notebooks/{folder}/index.html{anchor}">'
            f'<span class="gl-tile-head"><span class="gl-num">{i}</span>'
            f'<span class="gl-name">{html.escape(name)}</span></span>'
            f'<span class="gl-desc">{html.escape(desc)}</span>'
            f'<span class="gl-meta"><span>{n} notebooks</span><code>{module}</code></span></a>'
        )  # tile
    (GEN / "tiles.md").write_text('```{=html}\n<div class="gl-tiles">\n' + "\n".join(tiles) + "\n</div>\n```\n")  # tiles
    total = sum(counts)  # all notebooks
    (GEN / "stats.md").write_text(
        '```{=html}\n<div class="gl-stats">'
        f"<span><strong>{total}</strong>tutorial notebooks</span>"
        "<span><strong>9</strong>chapters</span>"
        "<span><strong>1</strong>import</span></div>\n```\n"
    )  # totals
    return total


def fix_links(text, source):
    """Point repository-relative links in a Markdown file at GitHub."""
    base = posixpath.dirname(source.relative_to(REPO).as_posix())  # folder of the source file
    local = {"docs/USER_GUIDE.md": "user_guide.qmd", "docs/API_REFERENCE.md": "api_reference.qmd"}  # pages on this site

    def target(path, image):
        if re.match(r"^(https?:|mailto:|#)", path):  # absolute or in-page
            return path
        path, _, frag = path.partition("#")  # keep anchors
        full = posixpath.normpath(posixpath.join(base, path))  # path from the repository root
        if full in local:  # page on this site
            return local[full] + (f"#{frag}" if frag else "")
        return (RAW if image else BLOB) + full + (f"#{frag}" if frag else "")  # GitHub

    text = re.sub(r"(!?)\[([^\]]*)\]\(([^)\s]+)\)", lambda m: f"{m[1]}[{m[2]}]({target(m[3], bool(m[1]))})", text)  # Markdown links
    return re.sub(r'src="([^"]+)"', lambda m: f'src="{target(m[1], True)}"', text)  # HTML images


def write_page(name, title, body, source):
    """A .qmd page from Markdown, without citation or dollar-math parsing."""
    front = f'---\ntitle: "{title}"\nfrom: markdown-citations-tex_math_dollars\n---\n\n'  # front matter
    (HOME / name).write_text(front + fix_links(body, source))  # page


def readme_sections(names):
    """The README sections with the given level-2 headings, in order."""
    text = (REPO / "README.md").read_text()  # README
    parts = re.split(r"(?m)^(?=## )", text)  # level-2 sections
    return "\n".join(p for p in parts if p.splitlines()[0][3:].strip() in names)  # selected


def drop_h1(path):
    """A Markdown file without its first-level title."""
    return re.sub(r"\A# .*\n+", "", path.read_text(), count=1)  # body


def main():
    counts = notebook_counts()  # notebooks per chapter
    total = write_generated(counts)  # tiles and totals
    print("notebooks per chapter:", dict(zip([c[0] for c in CHAPTERS], counts)), "total", total)  # report

    write_page("getting_started.qmd", "Getting Started",
               readme_sections({"Installation", "Quick start", "GPU support", "Distributed computing"}), REPO / "README.md")  # install and quick start
    write_page("user_guide.qmd", "User Guide", drop_h1(DOCS / "USER_GUIDE.md"), DOCS / "USER_GUIDE.md")  # full guide
    write_page("api_reference.qmd", "API Reference", drop_h1(DOCS / "API_REFERENCE.md"), DOCS / "API_REFERENCE.md")  # API

    subprocess.run(["quarto", "render"], cwd=HOME, check=True)  # render to _site/

    site = HOME / "_site"  # rendered site
    for item in site.iterdir():  # publish next to docs/notebooks/
        dest = DOCS / item.name  # destination
        if item.is_dir():  # site_libs, assets
            shutil.rmtree(dest, ignore_errors=True)  # replace the previous build
            shutil.copytree(item, dest)  # copy
        else:  # pages and search index
            shutil.copy2(item, dest)  # copy
    print("published", sorted(p.name for p in site.iterdir()), "to", DOCS)  # report


if __name__ == "__main__":
    main()
