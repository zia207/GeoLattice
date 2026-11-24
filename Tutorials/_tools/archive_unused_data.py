"""Move data folders that no tutorial notebook uses into ``archive_unused_data/`` (and back).

A folder qualifies when every file under it is ``not used`` in ``Tutorials/DATA_USAGE.csv``;
only the topmost such folder is moved. Two kinds of folder are kept in place even then:

- parts of a multi-file product that is used (a folder with ``product.xml``, ``manifest.safe``
  or ``*_MTL.txt`` above it): readers open ``imagery/``, ``metadata/calibration/`` and the like
  themselves, so the notebooks never name those files;
- folders whose name appears as a string in a notebook of the same chapter (anywhere for the
  repository ``data/`` folder), such as ``CACHE / "slc"`` or ``"preprocessed"``: caches and
  inputs reached through paths built at run time.

Folders keep their path relative to the repository, so ``data/Groundwater`` goes to ``archive_unused_data/data/Groundwater``. Every move is
recorded in ``archive_unused_data/MANIFEST.csv``.

    python Tutorials/_tools/archive_unused_data.py --dry-run   # list what would move
    python Tutorials/_tools/archive_unused_data.py             # move
    python Tutorials/_tools/archive_unused_data.py --restore   # move everything back
"""

import csv
import os
import sys
from datetime import datetime
from pathlib import Path, PurePosixPath

REPO = Path(__file__).resolve().parents[2]
ARCHIVE = REPO / "archive_unused_data"
MANIFEST = ARCHIVE / "MANIFEST.csv"
USAGE = REPO / "Tutorials" / "DATA_USAGE.csv"


def candidates():
    """Topmost folders whose files are all unused, with file count and size."""
    rows = list(csv.DictReader(open(USAGE)))
    roots = {r["data_root"] for r in rows}
    stat = {}
    for r in rows:
        for par in PurePosixPath(r["path"]).parents:
            s = str(par)
            if s == ".":
                break
            a = stat.setdefault(s, [0, 0, 0])
            a[0] += 1
            a[1] += r["status"] == "not used"
            a[2] += int(r["size_bytes"])
            if s in roots:
                break
    full = sorted((k for k, (n, u, _) in stat.items() if n == u and k not in roots), key=len)
    top = []
    for c in full:
        if not any(c.startswith(t + "/") for t in top):
            top.append(c)
    return [(t, stat[t][0], stat[t][2]) for t in top]


MARKERS = ("product.xml", "manifest.safe", "scenes.json", "request.json")  # products and download caches


def _is_product(d):
    try:
        names = os.listdir(d)
    except OSError:
        return False
    return any(n in MARKERS or n.endswith(("_MTL.txt", "_MTL.xml")) for n in names)


def _notebook_words():
    """String literals (and their path parts) per chapter, and over all notebooks."""
    import json
    import re

    per, every = {}, set()
    for nb in (REPO / "Tutorials" / "notebooks").rglob("*.ipynb"):
        if ".ipynb_checkpoints" in nb.parts:
            continue
        src = "".join("".join(c["source"]) for c in json.loads(nb.read_text())["cells"] if c["cell_type"] == "code")
        w = set(re.findall(r"""["']([^"'\n]{2,120})["']""", src))
        w |= {p for x in list(w) for p in x.split("/") if p}
        per.setdefault(nb.relative_to(REPO / "Tutorials" / "notebooks").parts[0], set()).update(w)
        every |= w
    return per, every


def protected(todo):
    """Drop folders that are parts of used products or that notebooks name in their code."""
    used = {r["path"] for r in csv.DictReader(open(USAGE)) if r["status"] != "not used"}
    per, every = _notebook_words()
    keep = []
    for rel, n, size in todo:
        o = PurePosixPath(rel)
        in_product = any(_is_product(REPO / a) and any(u.startswith(str(a) + "/") for u in used)
                         for a in list(o.parents)[:4] if str(a) != ".")
        words = per.get(o.parts[2], set()) if o.parts[0] == "Tutorials" else every
        named = len(o.name) >= 3 and o.name in words
        if not (in_product or named):
            keep.append((rel, n, size))
    return keep


def move(dry_run=False):
    todo = protected(candidates())
    ARCHIVE.mkdir(exist_ok=True)
    if os.stat(REPO).st_dev != os.stat(ARCHIVE).st_dev:
        sys.exit("archive is on another file system; refusing to copy instead of rename")
    new = not MANIFEST.exists()
    moved = 0
    with open(MANIFEST, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["original", "archived", "files", "bytes", "moved_at"])
        for rel, n, size in todo:
            src, dst = REPO / rel, ARCHIVE / rel
            if not src.exists():
                continue
            if dry_run:
                print(f"would move {rel} ({n} files, {size / 2**30:.2f} GB)")
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                sys.exit(f"target exists, stopping: {dst}")
            os.rename(src, dst)  # same file system: an instant rename, nothing is copied
            w.writerow([rel, str(dst.relative_to(REPO)), n, size, datetime.now().isoformat(timespec="seconds")])
            moved += 1
    print(f"{'would move' if dry_run else 'moved'} {moved if not dry_run else len(todo)} folders, "
          f"{sum(s for _, _, s in todo) / 2**30:.1f} GB")


def current_path(original):
    """Where an original path lives now that the data folders moved into ``Data/`` (``Data/MOVE_MANIFEST.csv``)."""
    moves = REPO / "Data" / "MOVE_MANIFEST.csv"
    pairs = [(m["original"], m["moved_to"]) for m in csv.DictReader(open(moves))] if moves.exists() else []
    for old, new in sorted(pairs, key=lambda x: len(x[0]), reverse=True):
        if original == old or original.startswith(old + "/"):
            return new + original[len(old):]
    if original.startswith("data/"):  # a whole repository dataset archived before the move
        return "Data/00_unassigned/" + original[len("data/"):]
    return original


def restore():
    rows = list(csv.DictReader(open(MANIFEST)))
    back = 0
    for r in reversed(rows):
        src, dst = REPO / r["archived"], REPO / current_path(r["original"])
        if src.exists() and not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.rename(src, dst)
            back += 1
    print(f"restored {back} folders")


if __name__ == "__main__":
    if "--restore" in sys.argv:
        restore()
    else:
        move(dry_run="--dry-run" in sys.argv)
