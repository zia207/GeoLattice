"""Build docs/API_REFERENCE.md from the installed ``geolattice`` package.

The reference follows the layout of the GeoPandas API reference: one section per
area of the library, sub-sections for the modules users import from, and a table
of every public class and function with the first sentence of its docstring.
Classes also list their public methods.

Each object is listed once, under the shortest public import path that reaches
it. Names a subpackage exports directly are grouped by the module that defines
them; submodules a subpackage exports (``geolattice.rsense.sar``,
``geolattice.spatial_analysis.segregation``, ...) get their own sub-sections.

    python docs/make_api_reference.py            # writes docs/API_REFERENCE.md
"""

from __future__ import annotations

import importlib
import inspect
import re
import warnings
from collections import OrderedDict
from pathlib import Path

warnings.filterwarnings("ignore")

import geolattice  # noqa: E402

OUT = Path(__file__).resolve().parent / "API_REFERENCE.md"

# (section title, top-level import, one-line description). Order follows the README.
SECTIONS = [
    ("Top-level package", "geolattice", "Readers, compute backends and configuration available after `import geolattice`."),
    ("Coordinate reference systems", "geolattice.crs", "Search EPSG and ESRI codes, define custom projections, choose datum transformations."),
    ("Core types", "geolattice.core", "Spatial data containers, bounds and metadata shared by the other modules."),
    ("Vector data", "geolattice.vector", "Read and write vector formats, cloud-native vector data, geoprocessing, geometry tools, vector analysis and attribute tables."),
    ("Raster data", "geolattice.raster", "Read, write and reproject rasters, and basic raster operations."),
    ("Spatial analysis", "geolattice.spatial_analysis", "Terrain analysis, surface hydrology, multidimensional rasters, multi-criteria suitability, network analysis, segregation analysis, spatial causal inference and vector analysis."),
    ("Spatial statistics", "geolattice.spatial_statistics", "Spatial weights, ESDA, geographically weighted models, point patterns, spatial regression, interpolation, sampling and Bayesian disease mapping."),
    ("Geostatistics", "geolattice.geostatistics", "Variograms, kriging, simulation, cross-validation and GSLIB-style helpers, on CPU, GPU (CuPy) or JAX."),
    ("Spatial databases", "geolattice.geolatticedb", "One API for PostGIS, SpatiaLite, GeoPackage and DuckDB spatial."),
    ("Web and cloud GIS", "geolattice.web", "Cloud-Optimized GeoTIFF, GeoParquet, XYZ tiles, tile and OGC API Features servers, and cloud storage access."),
    ("Remote sensing", "geolattice.rsense", "Data access, optical, SAR, InSAR, lidar, hyperspectral, UAV, time series, fusion, texture, super-resolution, classification and deep learning."),
    ("Visualization", "geolattice.viz", "Static maps, choropleths, raster displays and interactive maps."),
    ("Distributed and GPU computing", "geolattice.distributed", "One API for vector and raster processing, spatial statistics, networks, hydrology and machine learning on all CPU cores, a Dask cluster (local, multi-GPU, SLURM, PBS, Kubernetes), Spark with Apache Sedona, or CUDA GPUs."),
    ("Compute backends", "geolattice.compute", "CPU and GPU execution backends."),
    ("WhiteboxTools", "geolattice.whitebox", "Terrain and hydrology tools through WhiteboxTools."),
    ("Networks, point clouds and STAC", None, "Small helper modules."),
    ("Installation options", "geolattice.install_options", "The six installation options and install hints for optional dependencies."),
]
SMALL_MODULES = ["geolattice.network", "geolattice.pointcloud", "geolattice.stac"]
# Namespaces that only repeat another one.
ALIASES = {
    "geolattice.analysis": "alias of `geolattice.spatial_analysis` (also re-exports geostatistics and spatial statistics)",
    "geolattice.spatial_bd": "the package that holds `geolattice.geolatticedb`; `geolattice.spatial_bd.geolatticedb` and `geolattice.geolatticedb` are the same module",
    "geolattice.spatialcausal": "deprecated alias of `geolattice.spatial_analysis.spatialcausal` (emits a `DeprecationWarning`)",
    "geolattice.geostatistics.pygstat": "former name of `geolattice.geostatistics`, kept as an attribute alias",
}
SKIP_NAMESPACES = set(ALIASES) | {"geolattice.io", "geolattice.utils"}
MAX_DEPTH = 3


def signature_text(obj) -> str:
    try:
        sig = str(inspect.signature(obj))
    except (TypeError, ValueError):
        return ""
    sig = re.sub(r"\s+", " ", sig)
    if len(sig) > 120:
        sig = sig[:117].rsplit(",", 1)[0] + ", ...)"
    return f"`{obj.__name__}{sig}`"


def summary(obj) -> str:
    """First sentence of the docstring, as one line of Markdown.

    Without a docstring: a class's ``__init__`` docstring, otherwise the call signature.
    """
    doc = inspect.getdoc(obj) or ""
    if not doc and inspect.isclass(obj) and obj.__init__ is not object.__init__:
        doc = inspect.getdoc(obj.__init__) or ""
        if doc == inspect.getdoc(object.__init__):
            doc = ""
    if not doc:
        sig = signature_text(obj.__init__ if inspect.isclass(obj) else obj)
        if inspect.isclass(obj) and sig:
            sig = sig.replace("`__init__(self, ", f"`{obj.__name__}(").replace("`__init__(self)", f"`{obj.__name__}()")
        return f"Signature: {sig}." if sig else "No description."
    para = doc.strip().split("\n\n")[0].replace("\n", " ")
    para = re.sub(r"``([^`]+)``", r"`\1`", para)  # reST literals -> Markdown code
    para = re.sub(r":(?:func|class|meth|mod|attr|data|obj):`~?([^`]+)`", r"`\1`", para)
    para = re.sub(r"\s+", " ", para).strip()
    # house style: no em or en dashes (a spaced dash becomes a comma, Clark–Evans becomes Clark-Evans)
    para = re.sub(r"\s+[—–]\s+", ", ", para).replace("—", "-").replace("–", "-")
    # first sentence: a period followed by a space and a capital letter or the end
    match = re.search(r"^(.+?[.!?])(?:\s+[A-Z(`]|$)", para)
    text = match.group(1) if match else para
    if len(text) > 240:
        text = text[:237].rsplit(" ", 1)[0] + "..."
    return text.replace("|", "\\|") or "No description."


def methods_of(cls) -> list[str]:
    names = []
    for name, member in inspect.getmembers(cls):
        if name.startswith("_"):
            continue
        if inspect.isfunction(member) or inspect.ismethod(member) or isinstance(member, property):
            if name in getattr(object, "__dict__", {}):
                continue
            names.append(name)
    return names


def public_names(module) -> list[str]:
    names = getattr(module, "__all__", None)
    if names is None:
        names = [n for n in dir(module) if not n.startswith("_")]
    return list(dict.fromkeys(names))


def module_title(module) -> str:
    doc = inspect.getdoc(module) or ""
    first = doc.strip().split("\n")[0].strip().rstrip(".")
    first = re.sub(r"``([^`]+)``", r"`\1`", first)
    first = re.sub(r"\s+[—–]\s+", ", ", first).replace("—", "-").replace("–", "-")
    return first if 0 < len(first) <= 110 else ""


def collect(root: str, seen: set[int]) -> "OrderedDict[str, dict]":
    """Groups for one section: {group key: {"title": ..., "items": [(name, obj, path)]}}."""
    groups: "OrderedDict[str, dict]" = OrderedDict()

    def add(key, title, item):
        groups.setdefault(key, {"title": title, "items": []})["items"].append(item)

    def walk(path: str, depth: int):
        if path in SKIP_NAMESPACES or depth > MAX_DEPTH:
            return
        try:
            module = importlib.import_module(path)
        except Exception:
            return
        submodules = []
        for name in public_names(module):
            obj = getattr(module, name, None)
            if obj is None:
                continue
            if inspect.ismodule(obj):
                if obj.__name__.startswith("geolattice"):
                    submodules.append(f"{path}.{name}")
                continue
            if not (inspect.isclass(obj) or inspect.isfunction(obj) or inspect.isbuiltin(obj)):
                continue
            defining = getattr(obj, "__module__", "") or ""
            if not defining.startswith("geolattice"):
                continue  # re-exported third-party or standard-library objects (typing.Any, Path, ...)
            if not hasattr(module, "__all__") and defining != module.__name__:
                continue  # without __all__, list only what the module defines
            if id(obj) in seen:
                continue
            seen.add(id(obj))
            if path == root:  # group names exported at the top by the submodule that defines them
                depth_root = len(root.split("."))
                key = ".".join(defining.split(".")[:depth_root + 1]) if defining.startswith(root + ".") else root
                try:
                    title = module_title(importlib.import_module(key)) if key != root else ""
                except Exception:
                    title = ""
            else:
                key, title = path, module_title(module)
            add(key, title, (name, obj, path))
        # submodules a package documents without star-exporting them (``__submodules__``)
        for name in getattr(module, "__submodules__", ()):
            if f"{path}.{name}" not in submodules:
                submodules.append(f"{path}.{name}")
        for sub in submodules:
            walk(sub, depth + 1)

    walk(root, 0)
    return groups


def anchor(text: str) -> str:
    text = re.sub(r"[^\w\- ]", "", text.lower()).strip()
    return re.sub(r" ", "-", text)


def render_group(key: str, group: dict, root: str) -> list[str]:
    lines = []
    if key == root:
        heading = "Functions and classes" if root != "geolattice" else "Exports"
    else:
        heading = f"`{key}`"
    lines.append(f"### {heading}")
    lines.append("")
    if group["title"]:
        lines += [group["title"] + ".", ""]
    classes = [(n, o, p) for n, o, p in group["items"] if inspect.isclass(o)]
    funcs = [(n, o, p) for n, o, p in group["items"] if not inspect.isclass(o)]
    for label, items in (("Classes", classes), ("Functions", funcs)):
        if not items:
            continue
        lines += [f"**{label}**", "", "| Name | Description |", "|---|---|"]
        for name, obj, path in sorted(items, key=lambda t: t[0].lower()):
            desc = summary(obj)
            if inspect.isclass(obj):
                methods = methods_of(obj)
                if methods:
                    desc += " Methods: " + ", ".join(f"`{m}`" for m in methods) + "."
            lines.append(f"| `{path}.{name}` | {desc} |")
        lines.append("")
    return lines


def main():
    seen: set[int] = set()
    body, toc, counts = [], [], {}
    for title, root, blurb in SECTIONS:
        roots = SMALL_MODULES if root is None else [root]
        section_lines = [f"## {title}", "", blurb, ""]
        if root and root != "geolattice":
            section_lines += [f"`import {root}`", ""]
        n_items = 0
        for r in roots:
            groups = collect(r, seen) if r != "geolattice" else collect_top(seen)
            for key, group in groups.items():
                section_lines += render_group(key, group, r)
                n_items += len(group["items"])
        if n_items == 0:
            continue
        counts[title] = n_items
        toc.append(f"- [{title}](#{anchor(title)}) ({n_items})")
        body += section_lines
    head = [
        "# GeoLattice API reference",
        "",
        f"This page lists the public classes and functions of GeoLattice {geolattice.__version__}, grouped by area and by the module "
        "you import them from, with the first sentence of each docstring. Use `help(name)` in Python, or the "
        "[user guide](USER_GUIDE.md), for parameters and worked examples. Every subpackage is available after "
        "`import geolattice` (for example `geolattice.raster.clip`), and each object is listed once, under its shortest public "
        "import path.",
        "",
        f"The page is generated by `docs/make_api_reference.py` from the installed package ({sum(counts.values()):,} "
        "classes and functions). Rebuild it after changing the public API:",
        "",
        "```bash",
        "python docs/make_api_reference.py",
        "```",
        "",
        "## Contents",
        "",
        *toc,
        "- [Aliases](#aliases)",
        "",
    ]
    tail = ["## Aliases", "", "These namespaces repeat objects listed above and are not listed again.", "",
            "| Namespace | Same as |", "|---|---|"]
    tail += [f"| `{k}` | {v} |" for k, v in ALIASES.items()]
    OUT.write_text("\n".join(head + body + tail) + "\n")
    print(f"wrote {OUT} ({sum(counts.values())} entries)")
    for k, v in counts.items():
        print(f"  {v:5d}  {k}")


def collect_top(seen):
    """Names exported by ``geolattice`` itself (not its subpackages)."""
    groups = OrderedDict()
    for name in public_names(geolattice):
        obj = getattr(geolattice, name, None)
        if obj is None or inspect.ismodule(obj) or not (inspect.isclass(obj) or inspect.isfunction(obj)):
            continue
        seen.add(id(obj))
        groups.setdefault("geolattice", {"title": "", "items": []})["items"].append((name, obj, "geolattice"))
    return groups


if __name__ == "__main__":
    main()
