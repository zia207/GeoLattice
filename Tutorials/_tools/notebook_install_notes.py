"""Tell each tutorial notebook's reader which GeoLattice installation option it needs.

The option is the smallest of ``geolattice[core]``, ``[analysis]``, ``[statistics]``, ``[geostats]``,
``[remote-sensing]`` and ``[all]`` (see ``geolattice.install_options``) that covers every module the
notebook imports: its ``geolattice`` subpackages and its third-party libraries. Imports inside ``try:``
or ``if`` blocks (``if USE_GPU: import cupy``) are optional and do not count, nor do helper modules
that live in the notebook's chapter folder. Libraries that no option installs are named with their
own ``pip install`` command, or with the GeoLattice extra that provides them.

The note is one paragraph that starts with ``**Installation.**``. It is appended to the notebook's
Setup markdown cell, or placed in its own markdown cell before the first code cell when there is
no Setup cell. Running the script again replaces the note, so it stays in step with the imports.

    python Tutorials/_tools/notebook_install_notes.py --dry-run   # list the option per notebook
    python Tutorials/_tools/notebook_install_notes.py             # write the notes
"""

import ast
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
NB_ROOT = REPO / "Tutorials" / "notebooks"
OPTIONS = ["core", "analysis", "statistics", "geostats", "remote-sensing", "all"]
DESCRIPTION = {
    "core": "vector, raster, CRS and visualization",
    "analysis": "core plus spatial analysis",
    "statistics": "core, spatial analysis and spatial statistics",
    "geostats": "core, spatial analysis, spatial statistics and geostatistics",
    "remote-sensing": "core, spatial analysis, spatial statistics, geostatistics and remote sensing",
    "all": "every GeoLattice module",
}
MARK = "**Installation.**"

# geolattice subpackage -> smallest option
GEOLATTICE = {
    "": "core", "api": "core", "core": "core", "crs": "core", "io": "core", "raster": "core", "vector": "core",
    "viz": "core", "compute": "core", "install_options": "core",
    "analysis": "analysis", "network": "analysis", "spatial_analysis": "analysis",
    "spatial_analysis.spatialcausal": "statistics", "spatialcausal": "statistics", "spatial_statistics": "statistics",
    "geostatistics": "geostats",
    "rsense": "remote-sensing", "pointcloud": "remote-sensing",
    "web": "all", "geolatticedb": "all", "spatial_bd": "all",
}

# geolattice modules whose models come from an extra outside every option
GEOLATTICE_EXTRA = {"rsense.foundation_models": "rsense-foundation"}

# third-party import name -> smallest option that installs it (from pyproject.toml extras)
LIBS = {
    "core": "numpy pandas scipy shapely osgeo pyproj matplotlib mpl_toolkits geopandas folium branca contextily "
            "mapclassify plotly xarray pyvista rasterio fiona affine PIL xyzservices pyogrio pyarrow",
    "analysis": "netCDF4 h5netcdf zarr dask networkx osmnx libpysal sklearn whitebox",
    "statistics": "joblib",
    "geostats": "",
    "remote-sensing": "rioxarray skimage requests pystac_client pystac tqdm planetary_computer odc earthaccess h5py "
                      "laspy CSF cv2 pyodm gdown xgboost leafmap asf_search sarsen prosail",
    "all": "torch torchvision torchgeo segmentation_models_pytorch timm lightning shap ee xee streamlit "
           "starlette uvicorn httpx duckdb s3fs gcsfs adlfs sqlalchemy psycopg geoalchemy2 pymc arviz "
           "lightgbm pygam optuna jax jaxlib sliderule hyp3_sdk snaphu",
}
LIB_OPTION = {name: opt for opt, names in LIBS.items() for name in names.split()}
# Jupyter itself, not something GeoLattice installs
IGNORE = {"IPython", "ipykernel", "jupyter", "ipywidgets", "nbformat"}
# import name -> GeoLattice extra (outside every option) that installs it
EXTRA_FOR = {"cupy": "geostatistics-gpu", "tensorflow": "geostatistics-tensorflow", "h2o": "geostatistics-h2o",
             "openeo": "rsense-biopar", "segment_anything": "rsense-uav-dl", "ultralytics": "rsense-uav-dl",
             "open_clip": "rsense-vision", "samgeo": "rsense-vision",
             "terratorch": "rsense-foundation", "transformers": "rsense-foundation"}
# import name -> pip name, for libraries outside every option
PIP_NAME = {
    "skfuzzy": "scikit-fuzzy", "umap": "umap-learn", "yaml": "pyyaml", "hapi": "hitran-api", "sedona": "apache-sedona",
    "snowflake": "snowflake-connector-python", "google": "google-cloud-bigquery", "segregation": "segregation",
    "SALib": "SALib", "pyTMD": "pyTMD",
    "geojson_pydantic": "geojson-pydantic", "huggingface_hub": "huggingface_hub", "ipyleaflet": "ipyleaflet",
    "aurora": "microsoft-aurora", "sam2": "sam2",
}


def _optional(tree):
    """Import nodes inside a try or if block: they run only when the library is wanted or present."""
    inside = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Try, ast.If)):
            for child in node.body + getattr(node, "orelse", []):
                inside |= {id(n) for n in ast.walk(child) if isinstance(n, (ast.Import, ast.ImportFrom))}
    return inside


def requirements(nb_path, cells):
    """(option, sorted extra pip names) for one notebook."""
    chapter = NB_ROOT / nb_path.relative_to(NB_ROOT).parts[0]
    local = {p.stem for p in chapter.rglob("*.py")} | {p.name for p in nb_path.parent.iterdir() if p.is_dir()}
    std = set(sys.stdlib_module_names)
    needed, extra = {"core"}, set()
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        src = "\n".join(ln for ln in "".join(cell["source"]).splitlines() if not ln.lstrip().startswith(("%", "!")))
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        optional = _optional(tree)
        for node in ast.walk(tree):
            if id(node) in optional:
                continue
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names = [node.module]
            else:
                continue
            for name in names:
                parts = name.split(".")
                if parts[0] == "geolattice":
                    sub = ".".join(parts[1:3]) if parts[1:3] == ["spatial_analysis", "spatialcausal"] else ".".join(parts[1:2])
                    needed.add(GEOLATTICE.get(sub, "core"))
                    if ".".join(parts[1:3]) in GEOLATTICE_EXTRA:  # modules whose models need an extra outside the options
                        extra.add(f'"geolattice[{GEOLATTICE_EXTRA[".".join(parts[1:3])]}]"')
                elif parts[0] in std or parts[0] in local or parts[0] in IGNORE:
                    continue
                elif parts[0] in LIB_OPTION:
                    needed.add(LIB_OPTION[parts[0]])
                elif parts[0] in EXTRA_FOR:
                    extra.add(f'"geolattice[{EXTRA_FOR[parts[0]]}]"')
                else:
                    extra.add(PIP_NAME.get(parts[0], parts[0]))
    return max(needed, key=OPTIONS.index), sorted(extra, key=lambda e: e.strip('"').lower())


def note(option, extra):
    number = OPTIONS.index(option) + 1
    text = (f'{MARK} This notebook needs installation option {number} ({DESCRIPTION[option]}): '
            f'`pip install "geolattice[{option}]"`. See "Installation options" in the GeoLattice README.')
    if extra:
        names = ", ".join(f"`{e.strip(chr(34))}`" for e in extra)
        text += f" It also uses {names}, which no GeoLattice option installs: `pip install {' '.join(extra)}`."
    return text


def _set_source(cell, text):
    lines = text.split("\n")
    cell["source"] = [ln + "\n" for ln in lines[:-1]] + ([lines[-1]] if lines[-1] else [])


def update(nb_path, dry):
    nb = json.loads(nb_path.read_text())
    cells = nb["cells"]
    option, extra = requirements(nb_path, cells)
    paragraph = note(option, extra)
    # drop an earlier note: its own cell, or the paragraph inside the Setup cell
    cells[:] = [c for c in cells if not c.get("metadata", {}).get("geolattice_install_note")]
    for c in cells:
        if c["cell_type"] == "markdown" and MARK in "".join(c["source"]):
            src = "".join(c["source"])
            _set_source(c, re.sub(r"\n*" + re.escape(MARK) + r"[^\n]*", "", src).rstrip("\n"))
    setup = next((c for c in cells if c["cell_type"] == "markdown"
                  and re.search(r"(?im)^#{2,4}\s*setup\b", "".join(c["source"]))), None)
    if setup is not None:
        _set_source(setup, "".join(setup["source"]).rstrip("\n") + "\n\n" + paragraph)
        where = "setup cell"
    else:
        first_code = next((i for i, c in enumerate(cells) if c["cell_type"] == "code"), len(cells))
        new = {"cell_type": "markdown", "metadata": {"geolattice_install_note": True}, "source": [paragraph]}
        if (nb.get("nbformat"), nb.get("nbformat_minor", 0)) >= (4, 5):
            new["id"] = "geolattice-install-note"  # cell ids are required from nbformat 4.5
        cells.insert(first_code, new)
        where = "new cell"
    if not dry:
        nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    return option, extra, where


def check():
    """Compare the LIBS table with pip resolution reports (``--check report_dir``): r311_<option>.json."""
    report_dir = Path(sys.argv[sys.argv.index("--check") + 1])
    dist = {"osgeo": "gdal", "mpl_toolkits": "matplotlib", "PIL": "pillow", "sklearn": "scikit-learn",
            "skimage": "scikit-image", "cv2": "opencv-python", "odc": "odc-stac", "CSF": "cloth-simulation-filter",
            "ee": "earthengine-api", "psycopg": "psycopg", "netCDF4": "netcdf4", "pystac_client": "pystac-client",
            "planetary_computer": "planetary-computer", "segmentation_models_pytorch": "segmentation-models-pytorch",
            "hyp3_sdk": "hyp3-sdk", "asf_search": "asf-search", "xyzservices": "xyzservices"}
    norm = lambda s: re.sub(r"[-_.]+", "-", s).lower()
    sets = {o: {norm(i["metadata"]["name"]) for i in json.loads((report_dir / f"r311_{o}.json").read_text())["install"]}
            for o in OPTIONS}
    bad = []
    for name, opt in LIB_OPTION.items():
        d = norm(dist.get(name, name))
        first = next((o for o in OPTIONS if d in sets[o]), None)
        if first != opt:
            bad.append((name, opt, first))
    print("LIBS table matches pip reports" if not bad else f"mismatches (import, table, pip): {bad}")


def main(dry):
    counts, extras = Counter(), Counter()
    for nb_path in sorted(p for p in NB_ROOT.rglob("*.ipynb") if ".ipynb_checkpoints" not in p.parts):
        option, extra, where = update(nb_path, dry)
        counts[option] += 1
        extras.update(extra)
        if dry:
            print(f"{option:15s} {where:10s} {nb_path.relative_to(NB_ROOT)}" + (f"  + {', '.join(extra)}" if extra else ""))
    print(("DRY RUN: " if dry else "") + f"{sum(counts.values())} notebooks:",
          ", ".join(f"{o} {counts[o]}" for o in OPTIONS))
    if extras:
        print("libraries outside every option:", ", ".join(f"{k} ({v})" for k, v in extras.most_common()))


if __name__ == "__main__":
    if "--check" in sys.argv:
        check()
    else:
        main("--dry-run" in sys.argv)
