"""Which files in the data folders are used by the tutorial notebooks, and which are not.

Builds on ``data_inventory.py`` (run first, or this script runs it): every file in the
repository ``data/`` folder and in the chapter ``data/`` folders under ``Tutorials/notebooks``
is marked as

- ``used``: listed in ``DATA_INVENTORY.csv`` (directly, by name or by glob pattern);
- ``used (companion)``: a sidecar of a used file (same name stem in the same folder, for example
  ``.dbf``/``.shx``/``.prj`` of a shapefile, ``.hdr`` of ENVI, ``.aux.xml``);
- ``used (folder)``: inside a folder that a notebook passes to a reader, lists with ``iterdir`` /
  ``glob`` / ``os.listdir``, or inside a container dataset (``.zarr``, ``.gdb``, ``.SAFE``);
- ``not used``: none of the above.

    python Tutorials/_tools/data_usage.py     # writes Tutorials/DATA_USAGE.{csv,md}
"""

import ast
import re
import csv
import json
from collections import defaultdict
from pathlib import Path

import data_inventory as inv

REPO, NB_ROOT, OUT = inv.REPO, inv.NB_ROOT, inv.OUT
CONTAINERS = (".zarr", ".gdb", ".safe", ".sen3", ".ipynb_checkpoints")
PATH_BUILDERS = {"Path", "PurePath", "str", "join", "joinpath", "exists", "is_dir", "is_file", "mkdir", "makedirs",
                 "print", "relative_to", "resolve", "with_suffix", "with_name", "chdir", "insert", "append", "rmtree",
                 "next", "sorted", "len", "range", "isdir", "isfile"}
SKIP_NAMES = {"Thumbs.db", ".DS_Store", "desktop.ini"}


def data_roots():
    """Data/<group>/ folders (and Data/08_remote_sensing/<section>/)."""
    data = REPO / "Data"
    roots = []
    for g in sorted(d for d in data.iterdir() if d.is_dir()):
        subs = [d for d in g.iterdir() if d.is_dir() and re.match(r"\d\d_\d\d_", d.name)]
        roots += subs if subs else [g]
    return roots


def folder_uses():
    """Directories that notebooks read as a whole: (reader arguments, iterdir/glob/listdir bases).
    Returns (dirs from any use, dirs passed directly to a reader or catalogue function)."""
    used, as_arg = set(), set()
    for nb_path in sorted(p for p in NB_ROOT.rglob("*.ipynb") if ".ipynb_checkpoints" not in p.parts):
        nb = json.loads(nb_path.read_text())
        scope = inv.Scope(nb_path.parent)
        for cell in nb.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            try:
                tree = ast.parse(inv.clean("".join(cell.get("source", []))))
            except SyntaxError:
                continue
            inv.refs_in_cell(tree, scope, "")  # updates the path variables
            for node in ast.walk(tree):
                cands, direct = [], []
                if isinstance(node, ast.Call):
                    f = node.func
                    fname = f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else "")
                    if isinstance(f, ast.Attribute) and f.attr in ("iterdir", "glob", "rglob", "walk"):
                        cands.append(f.value)
                    if fname not in PATH_BUILDERS:  # Path("data"), str(x), mkdir(...) build paths, they do not read
                        cands += list(node.args)
                        direct = list(node.args)
                for c in cands:
                    v = scope.value(c)
                    if not v or len(v) > 300 or "\n" in v:
                        continue
                    v = v.replace("@REPO", str(REPO))
                    bare = isinstance(c, ast.Constant) and "/" not in v  # e.g. set_index("data"): a name, not a path
                    for base in ([Path(v)] if Path(v).is_absolute() else [nb_path.parent / v] if bare else
                                 [nb_path.parent / v, REPO / v, REPO / "Data" / v]):
                        try:
                            if base.is_dir():
                                used.add(base.resolve())
                                if c in direct:
                                    as_arg.add(base.resolve())
                                break
                        except OSError:
                            pass
    return used, as_arg


def human(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024


def main():
    if not (OUT / "DATA_INVENTORY.csv").exists():
        inv.main()
    used_files = set()
    with open(OUT / "DATA_INVENTORY.csv") as f:
        for r in csv.DictReader(f):
            if r["status"].startswith("found"):
                used_files.add((REPO / r["resolved_path"]).resolve())
    used_dirs, as_arg = folder_uses()
    roots = data_roots()
    # a whole data root counts only when passed straight to a reader / catalogue function (build_catalog(DATA)),
    # not when it is merely listed or globbed
    root_set = {r.resolve() for r in roots}
    used_dirs = {d for d in used_dirs if any(str(d).startswith(str(r)) for r in root_set)
                 and (d not in root_set or d in as_arg)}
    stems = defaultdict(set)  # folder -> used name stems (for companions)
    for f in used_files:
        stems[f.parent].add(f.name.split(".")[0])

    rows = []
    for root in roots:
        for f in sorted(root.rglob("*")):
            if not f.is_file() or f.name in SKIP_NAMES or "__pycache__" in f.parts or ".ipynb_checkpoints" in f.parts:
                continue
            rf = f.resolve()
            container = next((p for p in rf.parents if p.suffix.lower() in CONTAINERS), None)
            if rf in used_files:
                status = "used"
            elif container is not None and (container in used_dirs or any(str(u).startswith(str(container)) for u in used_files)
                                            or any(str(container).startswith(str(d)) for d in used_dirs)):
                status = "used (folder)"
            elif rf.name.split(".")[0] in stems.get(rf.parent, ()):
                status = "used (companion)"
            elif any(str(rf).startswith(str(d) + "/") for d in used_dirs):
                status = "used (folder)"
            else:
                status = "not used"
            rel_root = root.relative_to(REPO)
            rows.append({"data_root": str(rel_root), "path": str(f.relative_to(REPO)),
                         "dataset": str(Path(*f.relative_to(root).parts[:1])) if len(f.relative_to(root).parts) > 1 else "(top level)",
                         "size_bytes": f.stat().st_size, "status": status})
    with open(OUT / "DATA_USAGE.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    write_markdown(rows)
    return rows


def write_markdown(rows):
    total = sum(r["size_bytes"] for r in rows)
    unused = [r for r in rows if r["status"] == "not used"]
    lines = ["# Data Folder Usage", "",
             "Every file under the repository `Data/` folder, "
             "checked against the data the notebooks read (`DATA_INVENTORY.csv`). Generated by "
             "`Tutorials/_tools/data_usage.py`; the full file list is `Tutorials/DATA_USAGE.csv`.", "",
             f"- Files: **{len(rows):,}** ({human(total)})",
             f"- Used by at least one notebook: **{len(rows) - len(unused):,}** "
             f"({human(total - sum(r['size_bytes'] for r in unused))})",
             f"- Not used by any notebook: **{len(unused):,}** ({human(sum(r['size_bytes'] for r in unused))})", "",
             "*Used* also counts companion files of a used file (shapefile `.dbf`/`.shx`/`.prj`, ENVI `.hdr`, "
             "`.aux.xml`) and files inside folders a notebook reads as a whole (band folders, `.zarr`, `.gdb`, "
             "`.SAFE`). The check reads the code without running it, so treat *not used* as a strong hint, and look "
             "before deleting anything.", "",
             "## Contents", "", "- [Summary by data folder](#summary-by-data-folder)",
             "- [Files not used by any notebook](#files-not-used-by-any-notebook)", "",
             "## Summary by data folder", ""]
    by_root = defaultdict(list)
    for r in rows:
        by_root[r["data_root"]].append(r)
    for root in sorted(by_root):
        rs = by_root[root]
        lines += [f"### `{root}/`", "", "| Dataset | Files | Size | Used files | Not used files | Not used size |",
                  "|---|---:|---:|---:|---:|---:|"]
        by_ds = defaultdict(list)
        for r in rs:
            by_ds[r["dataset"]].append(r)
        for ds in sorted(by_ds):
            d = by_ds[ds]
            nu = [r for r in d if r["status"] == "not used"]
            flag = " **(entirely unused)**" if len(nu) == len(d) else ""
            lines.append(f"| `{ds}`{flag} | {len(d):,} | {human(sum(r['size_bytes'] for r in d))} | "
                         f"{len(d) - len(nu):,} | {len(nu):,} | {human(sum(r['size_bytes'] for r in nu))} |")
        lines.append("")
    lines += ["## Files not used by any notebook", "",
              "Grouped by folder; folders with more than 8 unused files are summarised.", ""]
    by_folder = defaultdict(list)
    for r in unused:
        by_folder[str(Path(r["path"]).parent)].append(r)
    current_root = None
    for folder in sorted(by_folder):
        root = next(r["data_root"] for r in by_folder[folder])
        if root != current_root:
            current_root = root
            lines += [f"### `{root}/`", ""]
        fs = by_folder[folder]
        size = human(sum(r["size_bytes"] for r in fs))
        if len(fs) > 8:
            exts = sorted({Path(r["path"]).suffix or "(no ext)" for r in fs})
            lines.append(f"- `{folder}/`: {len(fs)} files, {size} ({', '.join(exts[:8])})")
        else:
            for r in fs:
                lines.append(f"- `{r['path']}` ({human(r['size_bytes'])})")
    (OUT / "DATA_USAGE.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    rows = main()
    unused = [r for r in rows if r["status"] == "not used"]
    print(len(rows), "files |", len(unused), "not used |", human(sum(r["size_bytes"] for r in unused)), "unused")
