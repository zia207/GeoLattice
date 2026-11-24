"""Write requirements.txt: every runtime dependency of GeoLattice.

The list is the dependency closure of ``geolattice[all]`` (installation option 6) from
pyproject.toml, with each package listed once at its strictest lower bound. pyproject.toml
reads ``[project].dependencies`` from requirements.txt, so the wheel and the sdist install
every package in it. Extras that need CUDA, a cluster, Java, or pins that conflict with
this set stay out (they are not part of ``geolattice[all]``).

    python tools/make_requirements.py          # rewrite requirements.txt
    python tools/make_requirements.py --check  # exit 1 if requirements.txt is out of date

Run it after changing an extra in pyproject.toml, then rebuild the wheel.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

try:
    import tomllib
except ImportError:  # Python < 3.11
    import tomli as tomllib

from packaging.requirements import Requirement
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "requirements.txt"

# The packages every part of GeoLattice imports (the base install before the extras existed).
BASE = ["numpy>=1.21.0", "scipy>=1.7.0", "pandas>=1.3.0", "shapely>=1.8.0", "gdal>=3.3.0", "pyproj>=3.2.0"]

# (heading, extras whose packages are listed under it); a package goes under the first match
GROUPS = [
    ("Core: arrays, tables, geometry, CRS, GDAL", ["base", "io", "viz"]),
    ("Spatial analysis", ["multidim", "network", "segregation", "segregation-dasymetric", "whitebox"]),
    ("Spatial statistics and spatial causal inference",
     ["spatial-statistics", "spatial-statistics-viz", "spatial-statistics-deep", "spatial-statistics-boosting",
      "spatial-statistics-bayes", "spatial-statistics-gam", "spatial-statistics-libpysal",
      "spatial-statistics-network", "spatialcausal", "spatialcausal-viz", "spatialcausal-causalml"]),
    ("Geostatistics", ["geostatistics", "geostatistics-plot", "geostatistics-jax", "geostatistics-pytorch",
                       "geostatistics-gam", "geostatistics-optuna"]),
    ("Remote sensing", ["rsense*"]),
    ("Web and cloud GIS, spatial databases", ["web", "web-cloud", "spatial-db", "spatial-db-postgis"]),
    ("Distributed computing and machine learning", ["distributed", "distributed-ml"]),
]

HEADER = """\
# GeoLattice runtime dependencies.
#
# This file is the single source of truth: pyproject.toml reads [project].dependencies
# from it, so installing the wheel or the sdist (pip install geolattice-<version>-*.whl or
# geolattice-<version>.tar.gz) installs every package listed here. It is the conflict-free set
# that "geolattice[all]" (installation option 6) resolves to. Regenerate it with
# "python tools/make_requirements.py" after changing an extra in pyproject.toml;
# tests/test_requirements.py checks that the two agree.
#
# Left out on purpose, install by name when needed (they need CUDA, a cluster, Java, or
# pin versions that conflict with this set):
#   geolattice[compute], geolattice[geostatistics-gpu], geolattice[rsense-gpu], geolattice[distributed-gpu]   CUDA GPUs
#   geolattice[distributed-hpc]       SLURM, PBS and Kubernetes clusters
#   geolattice[distributed-spark]     Apache Sedona and PySpark (Java 17+)
#   geolattice[geostatistics-h2o], geolattice[geostatistics-tensorflow]
#   geolattice[rsense-uav-dl], geolattice[rsense-vision], geolattice[rsense-foundation]   large vision models
#   geolattice[rsense-biopar]         needs pandas < 3
#   geolattice[dev]                   tests, linters, docs (see requirements-dev.txt)
#
# The "gdal" package compiles against the GDAL library already on the system and must
# match its version; install GDAL with conda or the system package manager first.
"""


def _extras() -> dict[str, list[str]]:
    return tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["optional-dependencies"]


def closure(extra: str, extras: dict[str, list[str]], seen: set[str] | None = None) -> list[tuple[str, str]]:
    """(extra, requirement) pairs that ``geolattice[extra]`` installs, following geolattice[...] references."""
    seen = set() if seen is None else seen
    if extra in seen:
        return []
    seen.add(extra)
    out = []
    for req in extras[extra]:
        m = re.match(r"geolattice\[(.+)\]", req)
        if m:
            for e in m.group(1).split(","):
                out += closure(e.strip(), extras, seen)
        else:
            out.append((extra, req))
    return out


def merged() -> dict[str, dict]:
    """One entry per package: name, extras, URL or strictest ``>=`` bound, other specifiers."""
    extras = _extras()
    pkgs: dict[str, dict] = {}
    for src, req in [("base", r) for r in BASE] + closure("all", extras):
        q = Requirement(req)
        key = q.name.lower().replace("_", "-")
        m = pkgs.setdefault(key, {"name": q.name, "extras": set(), "spec": {}, "url": None,
                                  "marker": q.marker, "srcs": []})
        m["extras"] |= q.extras
        m["srcs"].append(src)
        m["url"] = m["url"] or q.url
        for s in q.specifier:
            if s.operator == ">=":
                cur = m["spec"].get(">=")
                if cur is None or Version(s.version) > Version(cur):
                    m["spec"][">="] = s.version
            else:
                m["spec"][f"{s.operator}{s.version}"] = None
    return pkgs


def line(m: dict) -> str:
    ex = f"[{','.join(sorted(m['extras']))}]" if m["extras"] else ""
    if m["url"]:
        out = f"{m['name']}{ex} @ {m['url']}"
    else:
        specs = sorted((f">={v}" if k == ">=" else k) for k, v in m["spec"].items())
        out = m["name"] + ex + ",".join(specs)
    return out + (f"; {m['marker']}" if m["marker"] else "")


def render() -> str:
    def group_of(m: dict) -> str:
        for title, pats in GROUPS:
            for src in m["srcs"]:
                if any(src == p or (p.endswith("*") and src.startswith(p[:-1])) for p in pats):
                    return title
        return "Other"

    by: dict[str, list[str]] = {}
    for m in merged().values():
        by.setdefault(group_of(m), []).append(line(m))
    parts = [HEADER]
    for title in [g for g, _ in GROUPS] + ["Other"]:
        if title in by:
            parts.append(f"\n# {title}\n" + "\n".join(sorted(by[title], key=str.lower)) + "\n")
    return "".join(parts)


def main(argv: list[str]) -> int:
    text = render()
    if "--check" in argv:
        if OUT.read_text() != text:
            print("requirements.txt is out of date; run: python tools/make_requirements.py")
            return 1
        return 0
    OUT.write_text(text)
    print(f"wrote {OUT} ({sum(1 for l in text.splitlines() if l and not l.startswith('#'))} packages)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
