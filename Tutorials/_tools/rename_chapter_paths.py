"""Update paths after the chapters were renumbered.

    07_spatialDB       -> 06_spatialDB       (sections 07_0X_* -> 06_0X_*)
    08_web_cloud_gis   -> 07_web_cloud_gis   (notebooks 08_XX_* -> 07_XX_*)
    09_remote_sensing  -> 08_remote_sensing  (sections and notebooks 09_* -> 08_*)

A section or notebook name is rewritten only when the new name exists on disk, so prose such as
"notebook 09_05_01" and the data folders ``Data/00_shared/data_web_cloud_gis/08_01_outputs`` are left
alone. Code and markdown cells of notebooks are rewritten (outputs are not), and so are helper ``.py``
and ``.md`` files under ``Tutorials``, the live docs and ``Data/MOVE_MANIFEST.csv`` (``moved_to``).

    python Tutorials/_tools/rename_chapter_paths.py --dry-run
    python Tutorials/_tools/rename_chapter_paths.py
"""

import csv
import io
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
NB = REPO / "Tutorials" / "notebooks"
CHAPTERS = {"09_remote_sensing": "08_remote_sensing", "08_web_cloud_gis": "07_web_cloud_gis",
            "07_spatialDB": "06_spatialDB", "07_spatialBd": "06_spatialDB"}
SKIP = {"DATA_INVENTORY.md", "DATA_USAGE.md", "DATA_BY_CHAPTER.md", "GeoLattice_session_prompts_and_responses.md"}


def names(*roots, depth=3):
    """Names (and stems) of files and folders under the renamed chapters."""
    out = set()
    for root in roots:
        if not root.is_dir():
            continue
        for p in root.rglob("*"):
            if len(p.relative_to(root).parts) <= depth and ".ipynb_checkpoints" not in p.parts:
                out |= {p.name, p.stem}
    return out


RS = names(NB / "08_remote_sensing", REPO / "Data" / "08_remote_sensing")
WEB = names(NB / "07_web_cloud_gis", depth=1)
DB = names(NB / "06_spatialDB", REPO / "Data" / "06_spatialDB", depth=2)
DB_SECTIONS = {re.sub(r"\W+", "_", d.name.lower()): d.name for d in (NB / "06_spatialDB").iterdir() if d.is_dir()}
STATS = Counter()


def _count(rule, new):
    STATS[rule] += 1
    return new


def rewrite(t):
    t = re.sub("|".join(CHAPTERS), lambda m: _count("chapter", CHAPTERS[m.group(0)]), t)

    def rs(m):  # 09_02_optical -> 08_02_optical, 09_05_01_x -> 08_05_01_x
        new = "08_" + m.group(1)
        return _count("remote sensing name", new) if new in RS else m.group(0)
    t = re.sub(r"(?<![\w.])09_(\d\d_[A-Za-z0-9][\w\-]*)", rs, t)

    def web(m):  # 08_01_cloud_native_raster.ipynb -> 07_01_cloud_native_raster.ipynb
        new = "07_" + m.group(1)
        return _count("web notebook", new) if new in WEB and "08_" + m.group(1) not in RS else m.group(0)
    t = re.sub(r"(?<![\w.])08_(\d\d_[A-Za-z][\w\-]*)(?=\.ipynb)", web, t)

    def db(m):  # 07_02_Spatila_Database_Core -> 06_02_spatial_database_core
        key = "06_" + re.sub(r"\W+", "_", m.group(1).lower()).replace("spatila", "spatial").replace("foudation", "foundation")
        return _count("spatialDB section", DB_SECTIONS[key]) if key in DB_SECTIONS else m.group(0)
    t = re.sub(r"(?<![\w.])07_(0\d_(?:Foudation|Spatila_Database_Core|Advanced Spatial Databases|Applications and Deployment))", db, t)
    return t


def notebook(f, dry):
    nb = json.loads(f.read_text())
    changed = False
    for c in nb.get("cells", []):
        src = "".join(c.get("source", []))
        new = rewrite(src)
        if new != src:
            changed = True
            lines = new.split("\n")
            c["source"] = [ln + "\n" for ln in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    if changed and not dry:
        f.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    return changed


def text(f, dry):
    t = f.read_text()
    new = rewrite(t)
    if new != t and not dry:
        f.write_text(new)
    return new != t


def manifest(dry):
    f = REPO / "Data" / "MOVE_MANIFEST.csv"
    rows = list(csv.DictReader(open(f)))
    for r in rows:
        r["moved_to"] = rewrite(r["moved_to"])
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
    if not dry:
        f.write_text(buf.getvalue())


def targets():
    for f in sorted((REPO / "Tutorials").rglob("*")):
        if ".ipynb_checkpoints" in f.parts or f.name in SKIP or f.name == Path(__file__).name:
            continue
        if f.suffix in (".ipynb", ".py", ".md"):
            yield f
    for f in (REPO / "README.md", REPO / "Data" / "README.md", REPO / "docs" / "GeoLattice_modules_organization_and_rebuild.md"):
        if f.exists():
            yield f


def main(dry):
    files = [f for f in targets() if (notebook if f.suffix == ".ipynb" else text)(f, dry)]
    manifest(dry)
    print(("DRY RUN: " if dry else "") + f"{len(files)} files changed")
    for k, v in STATS.most_common():
        print(f"  {v:5d}  {k}")
    return files


if __name__ == "__main__":
    main("--dry-run" in sys.argv)
