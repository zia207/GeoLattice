"""Inventory of the data files used by every tutorial notebook.

Parses the code cells of each notebook (AST), follows simple path variables
(``DATA = Path("data")``, ``F = DATA / "x.tif"``, ``os.path.join(...)``), and
records file literals with data extensions, glob patterns and download URLs.
Each local reference is resolved against the notebook folder (and the repository
root) and checked for existence.

    python Tutorials/_tools/data_inventory.py      # writes Tutorials/DATA_INVENTORY.{csv,md}
"""

import ast
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
NB_ROOT = REPO / "Tutorials" / "notebooks"
OUT = REPO / "Tutorials"

EXT = {".tif", ".tiff", ".gpkg", ".shp", ".geojson", ".json", ".csv", ".parquet", ".geoparquet",
       ".nc", ".nc4", ".h5", ".he5", ".hdf", ".hdf5", ".zarr", ".las", ".laz", ".copc", ".npy", ".npz",
       ".mat", ".txt", ".xlsx", ".xls", ".gdb", ".zip", ".pmtiles", ".fgb", ".img", ".hdr", ".jp2",
       ".xml", ".kml", ".gml", ".dbf", ".asc", ".dem", ".grd", ".vrt", ".pkl", ".pickle", ".pt", ".pth",
       ".onnx", ".gz", ".tar", ".7z", ".sqlite", ".db", ".duckdb", ".feather", ".dat", ".bil", ".bsq",
       ".png", ".jpg", ".jpeg", ".data", ".header", ".mbtiles", ".osm", ".pbf", ".graphml", ".rds",
       ".safe", ".SAFE", ".tsv", ".xyz", ".ply", ".e57", ".ecw", ".sid", ".pts", ".obj"}
IMAGE_EXT = {".png", ".jpg", ".jpeg"}
URL_RE = re.compile(r"https?://[^\s'\"<>]+")
WRITE_HINT = re.compile(r"\b(to_file|to_csv|to_parquet|savefig|save|write_\w+|to_raster|to_netcdf|to_zarr|imwrite|dump)\s*\(")


def clean(src):
    """Drop IPython magics and shell lines so the cell parses."""
    out = []
    for line in src.splitlines():
        s = line.lstrip()
        out.append("" if s.startswith(("%", "!", "?")) else line)
    return "\n".join(out)


class Scope:
    def __init__(self, nb_dir):
        self.vars = {}  # name -> str path (relative to notebook dir, or absolute / URL)
        self.nb_dir = nb_dir

    def value(self, node):
        """Best effort evaluation of a path expression to a string, else None."""
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name):
            return self.vars.get(node.id)
        if isinstance(node, ast.JoinedStr):  # f-string: keep the literal parts, '*' for fields
            return "".join(v.value if isinstance(v, ast.Constant) else "*" for v in node.values)
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Div, ast.Add)):
            a, b = self.value(node.left), self.value(node.right)
            if a is not None and b is not None:
                return f"{a.rstrip('/')}/{b.lstrip('/')}" if isinstance(node.op, ast.Div) else a + b
            return None
        if isinstance(node, ast.Call):
            f = node.func
            fname = f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else "")
            if fname in ("cwd", "home") or fname.startswith("find_repo_root") or fname in ("find_root", "repo_root"):
                return "@REPO"  # notebooks climb from the working directory to the repository root
            if fname in ("Path", "PurePath", "str", "expanduser", "resolve", "absolute") and node.args:
                return self.value(node.args[0])
            if fname in ("resolve", "absolute", "expanduser") and isinstance(f, ast.Attribute):
                return self.value(f.value)
            if fname == "join" and isinstance(f, ast.Attribute) and isinstance(f.value, ast.Attribute):
                parts = [self.value(a) for a in node.args]
                return "/".join(parts) if all(p is not None for p in parts) else None
            if fname == "joinpath" and isinstance(f, ast.Attribute):
                parts = [self.value(f.value)] + [self.value(a) for a in node.args]
                return "/".join(parts) if all(p is not None for p in parts) else None
        if isinstance(node, ast.Attribute) and node.attr in ("parent", "parents"):
            v = self.value(node.value)
            if v and v.startswith("@"):
                return v  # ROOT = ROOT.parent while searching upwards: still the repository root
            return str(Path(v).parent) if v else None
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute) and node.value.attr == "parents":
            return "@REPO"
        if isinstance(node, ast.IfExp):
            return self.value(node.body)
        return None


def looks_like_data(s):
    if not s or len(s) > 400 or "\n" in s:
        return False
    if URL_RE.fullmatch(s):
        return True
    suf = Path(s.split("?")[0]).suffix
    return suf in EXT or suf.lower() in EXT


def refs_in_cell(tree, scope, src):
    """Yield (reference string, how) for one parsed cell, updating path variables."""
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            v = scope.value(node.value)
            if v is not None and ("/" in v or looks_like_data(v) or v in (".", "data") or v.startswith("@")):
                name = node.targets[0].id
                if not (name in scope.vars and scope.vars[name].startswith("@") and not v.startswith("@")):
                    scope.vars[name] = v
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ("glob", "rglob") \
                and node.args:
            base, pat = scope.value(node.func.value), scope.value(node.args[0])
            if pat:
                found.append((f"{base.rstrip('/')}/{pat}" if base is not None else f"@ANY/{pat}", "glob"))
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            v = scope.value(node)
            if v and looks_like_data(v):
                found.append((v, "path"))
        elif isinstance(node, (ast.Constant, ast.JoinedStr)):
            v = scope.value(node)
            if isinstance(v, str):
                for u in URL_RE.findall(v):
                    found.append((u.rstrip(").,;"), "url"))
                if looks_like_data(v) and not URL_RE.match(v):
                    found.append((v, "literal"))
    for name, v in scope.vars.items():  # variables holding data files count too
        if looks_like_data(v) and not URL_RE.match(v):
            found.append((v, "variable"))
    return found


def resolve(ref, nb_dir):
    """Absolute candidate path, and whether it exists (globs: any match)."""
    if URL_RE.match(ref):
        return ref, "url"
    ref = ref.replace("@REPO", str(REPO))
    if ref.startswith("@ANY/"):
        return Path(ref), "no"
    p = Path(ref).expanduser()
    cands = [p] if p.is_absolute() else [nb_dir / p, REPO / p, REPO / "Data" / p, nb_dir / "data" / p, nb_dir.parent / p]
    for c in cands:
        if any(ch in str(c) for ch in "*?["):
            try:
                if c.is_absolute() and next(Path("/").glob(str(c).lstrip("/")), None):
                    return c, "yes"
            except (ValueError, OSError, NotImplementedError):
                pass
        elif c.exists():
            return c.resolve(), "yes"
    return cands[0], "no"


DATA_DIRS = [REPO / "Data"]  # all tutorial data lives under Data/<group>/ (see Data/MOVE_MANIFEST.csv)
_INDEX = None


def by_name(name, nb_dir, prefer=()):
    """Files in the data folders whose name matches; the closest to the notebook first."""
    global _INDEX
    if _INDEX is None:
        _INDEX = defaultdict(list)
        for d in DATA_DIRS:
            for f in d.rglob("*"):
                _INDEX[f.name].append(f)
    hits = sorted(set(_INDEX.get(name, [])))
    rel = nb_dir.relative_to(NB_ROOT).parts if nb_dir != NB_ROOT else ()
    chapter = REPO / "Data" / rel[0] if rel else REPO / "Data"

    def distance(f):
        if any(str(f).startswith(str(d) + "/") for d in prefer):
            return -1  # inside a folder the notebook's own path variables point to
        if len(rel) > 1 and str(f).startswith(str(REPO / "Data" / rel[0] / rel[1]) + "/"):
            return 0
        if str(f).startswith(str(chapter) + "/"):
            return 1
        return 2 if str(f).startswith(str(REPO / "Data" / "00_shared") + "/") else 3
    return sorted(hits, key=distance)


def rel(p):
    try:
        return str(Path(p).resolve().relative_to(REPO))
    except (ValueError, OSError):
        return str(p)


def main():
    rows = []
    notebooks = sorted(p for p in NB_ROOT.rglob("*.ipynb") if ".ipynb_checkpoints" not in p.parts)
    for nb_path in notebooks:
        nb = json.loads(nb_path.read_text())
        scope = Scope(nb_path.parent)
        seen = set()
        words = set()  # short string literals: folder names the notebook mentions
        for cell in nb.get("cells", []):
            if cell.get("cell_type") == "code":
                for w in re.findall(r"""["']([\w.\- ]{3,60})["']""", "".join(cell.get("source", []))):
                    words.add(w)
        for cell in nb.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            src = clean("".join(cell.get("source", [])))
            try:
                tree = ast.parse(src)
            except SyntaxError:
                tree = ast.parse("")
                for u in URL_RE.findall(src):
                    if (u, "url") not in seen:
                        seen.add((u, "url"))
            writes = bool(WRITE_HINT.search(src))
            for ref, how in refs_in_cell(tree, scope, src):
                key = (ref, how if how in ("url", "glob") else "file")
                if key in seen:
                    continue
                seen.add(key)
                target, exists = resolve(ref, nb_path.parent)
                if ref != ref.strip() or " from " in ref:
                    continue  # prose in print statements
                if exists == "no" and how == "glob":
                    pat = Path(ref.replace("@ANY/", "")).name
                    literal = re.sub(r"[*?\[\]]", "", pat)
                    if len(literal) >= 4:
                        ch = nb_path.relative_to(NB_ROOT).parts
                        sec = REPO / "Data" / ch[0] / ch[1] if len(ch) > 2 else REPO / "Data" / ch[0]
                        roots = [nb_path.parent, sec, REPO / "Data" / ch[0], REPO / "Data" / "00_shared", REPO / "Data"]
                        matches = []
                        for root in roots:
                            if root.is_dir():
                                try:
                                    matches = [m for m in root.rglob(pat) if ".ipynb_checkpoints" not in m.parts][:41]
                                except ValueError:  # e.g. '**.xml', not a valid pathlib pattern
                                    matches = []
                                if matches:
                                    break
                        import os
                        common = Path(os.path.commonpath([str(m.parent) for m in matches])) if matches else None
                        named = [m for m in matches
                                 if any(part in words for part in m.parent.relative_to(common).parts)] if common else []
                        if named and len({m.parent for m in matches}) > 1:
                            matches = named  # keep folders the notebook names (e.g. tanager_file("agriculture"))
                        if 0 < len(matches) <= 40:
                            for m in matches:
                                if (str(m), "file") in seen:
                                    continue
                                seen.add((str(m), "file"))
                                rows.append({"chapter": nb_path.relative_to(NB_ROOT).parts[0], "notebook": rel(nb_path),
                                             "reference": ref.replace("@ANY/", "…/"), "kind": "glob",
                                             "resolved_path": rel(m), "exists": "yes (by pattern)", "cell_writes": ""})
                            continue
                    ref = ref.replace("@ANY/", "…/")
                if exists == "no" and how != "glob" and "*" not in ref:
                    prefer = [Path(v.replace("@REPO", str(REPO))) if v.startswith("@") or Path(v).is_absolute()
                              else nb_path.parent / v for v in scope.vars.values()
                              if not looks_like_data(v) and len(v) < 300 and "\n" not in v]
                    prefer = [d for d in prefer if d != REPO and d.is_dir()]
                    hits = by_name(Path(ref).name, nb_path.parent, prefer)
                    tail = Path(ref.replace("@REPO", "")).parts[-3:]  # partial paths must match their tail
                    hits = [h for h in hits if h.parts[-len(tail):] == tail] or ([] if len(tail) > 1 else hits)
                    if hits:
                        target, exists = hits[0], "yes (by name)"
                if Path(ref).suffix.lower() in IMAGE_EXT and not re.search(r"/[Dd]ata/", str(target)):
                    continue  # images outside data folders are figures the notebook writes
                if exists == "no" and (" " in ref.strip() or ref.count(":") > 1):
                    continue  # prose in print statements or labels, not a path
                rows.append({"chapter": nb_path.relative_to(NB_ROOT).parts[0], "notebook": rel(nb_path),
                             "reference": ref, "kind": "url" if exists == "url" else ("glob" if how == "glob" else "file"),
                             "resolved_path": target if exists == "url" else rel(target),
                             "exists": exists, "cell_writes": "yes" if writes else ""})
    # drop references that only appear as outputs: unresolved files in cells that write files
    keep = [r for r in rows if not (r["exists"] == "no" and r["cell_writes"] and r["kind"] == "file")]
    direct = {(r["notebook"], Path(str(r["resolved_path"])).name) for r in keep if r["exists"] == "yes"}
    out, seen_rows = [], set()
    for r in keep:
        if r["exists"] == "yes (by name)" and (r["notebook"], Path(str(r["resolved_path"])).name) in direct:
            continue
        k = (r["notebook"], str(r["resolved_path"])) if r["exists"] != "no" else (r["notebook"], r["reference"])
        if k in seen_rows:
            continue
        seen_rows.add(k)
        out.append(r)
    keep = out
    for r in keep:
        r["status"] = {"yes": "found", "yes (by name)": "found (by file name)", "yes (by pattern)": "found (by glob pattern)",
                       "url": "download URL",
                       "no": "not found (written or downloaded by the notebook, or missing)"}[r["exists"]]
    OUT.mkdir(exist_ok=True)
    with open(OUT / "DATA_INVENTORY.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["chapter", "notebook", "reference", "kind", "resolved_path", "status"])
        w.writeheader()
        for r in keep:
            w.writerow({k: r[k] for k in w.fieldnames})
    write_markdown(notebooks, keep)
    return notebooks, keep


def write_markdown(notebooks, rows):
    """Readable inventory: per notebook data list, a dataset summary and the unresolved references."""
    by_nb = defaultdict(list)
    for r in rows:
        by_nb[r["notebook"]].append(r)
    found = [r for r in rows if r["status"].startswith("found")]
    datasets = defaultdict(set)  # data folder -> notebooks
    sizes = {}
    for r in found:
        p = REPO / r["resolved_path"]
        parts = Path(r["resolved_path"]).parts
        # dataset = first folder below a data directory (data/<x>/ or .../data/<x>/)
        if parts[0] == "Data" and len(parts) > 2:  # Data/<group>/<dataset> or Data/08_remote_sensing/<section>/<dataset>
            depth = 4 if len(parts) > 3 and re.match(r"\d\d_\d\d_", parts[2]) else 3
            key = "/".join(parts[:min(depth, len(parts) - 1)])
        else:
            key = str(Path(r["resolved_path"]).parent)
        datasets[key].add(r["notebook"])
    lines = ["# Tutorial Data Inventory", "",
             "Data files read by the GeoLattice tutorial notebooks (`Tutorials/notebooks`), found by parsing every code "
             "cell. Paths are relative to the repository root. Generated by `Tutorials/_tools/data_inventory.py`; "
             "the full table is `Tutorials/DATA_INVENTORY.csv`.", "",
             f"- Notebooks scanned: **{len(notebooks)}**, notebooks that reference data: **{len(by_nb)}**",
             f"- Local data references found: **{len(found)}** in **{len(datasets)}** dataset folders",
             f"- Download URLs: **{sum(r['kind'] == 'url' for r in rows)}**",
             f"- Unresolved local references: **{sum(r['status'].startswith('not') for r in rows)}** "
             "(listed at the end; most are outputs or files fetched at run time)", "",
             "Status: *found* means the path resolves as written; *found (by file name)* means the notebook builds the "
             "path at run time (helper functions, loops) and a file of that name exists in a data folder; *found (by glob "
             "pattern)* means the notebook opens files with `glob` and these files match the pattern (it may open only "
             "some of them).", "",
             "## Contents", "", "- [Datasets and the notebooks that use them](#datasets-and-the-notebooks-that-use-them)",
             "- [Data by notebook](#data-by-notebook)", "- [Unresolved references](#unresolved-references)", "",
             "## Datasets and the notebooks that use them", "",
             "| Dataset folder | Notebooks | Used by |", "|---|---|---|"]
    for key in sorted(datasets):
        nbs = sorted(datasets[key])
        short = ", ".join(Path(n).stem.split("_", 3)[0] + "_" + Path(n).stem.split("_")[1] if Path(n).stem[:2].isdigit()
                          else Path(n).stem for n in nbs[:12])
        names = sorted({Path(n).stem for n in nbs})
        note = " *(outputs of another notebook)*" if "/outputs" in key else ""
        lines.append(f"| `{key}`{note} | {len(nbs)} | {', '.join(names[:8])}{' …' if len(names) > 8 else ''} |")
    lines += ["", "## Data by notebook", ""]
    chapter = None
    for nb_path in sorted(by_nb):
        ch = Path(nb_path).parts[2]
        if ch != chapter:
            chapter = ch
            lines += [f"### {ch}", ""]
        items = by_nb[nb_path]
        lines.append(f"**{Path(nb_path).relative_to('Tutorials/notebooks')}**")
        lines.append("")
        local = [r for r in items if r["status"].startswith("found")]
        groups = defaultdict(list)
        for r in local:
            groups[str(Path(r["resolved_path"]).parent)].append(r)
        for folder in sorted(groups):
            g = groups[folder]
            if len(g) > 6:
                exts = sorted({Path(r["resolved_path"]).suffix for r in g})
                lines.append(f"- `{folder}/` ({len(g)} files: {', '.join(exts)})")
            else:
                for r in sorted(g, key=lambda x: x["resolved_path"]):
                    tag = " *(by file name)*" if "by file" in r["status"] else (
                        " *(by glob pattern)*" if "glob" in r["status"] else "")
                    lines.append(f"- `{r['resolved_path']}`{tag}")
        for r in items:
            if r["kind"] == "url":
                lines.append(f"- download: <{r['reference']}>")
        lines.append("")
    lines += ["## Unresolved references", "",
              "References that point to no existing file: outputs the notebook writes, files it downloads at run "
              "time, glob patterns whose files are created during the run, or data that is missing.", "",
              "| Notebook | Reference |", "|---|---|"]
    for r in rows:
        if r["status"].startswith("not"):
            lines.append(f"| {Path(r['notebook']).relative_to('Tutorials/notebooks')} | `{r['reference']}` |")
    (OUT / "DATA_INVENTORY.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    nbs, rows = main()
    print(len(nbs), "notebooks |", len(rows), "references |",
          sum(r["exists"] == "yes" for r in rows), "local found |", sum(r["exists"] == "no" for r in rows),
          "local missing |", sum(r["kind"] == "url" for r in rows), "urls")
