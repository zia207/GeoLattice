"""Rewrite data paths after the move of ``data/`` and the chapter ``data/`` folders into ``Data/``.

Layout after the move (see ``Data/MOVE_MANIFEST.csv``):

- ``data/<dataset>``  ->  ``Data/<group>/<dataset>``; ``<group>`` is ``00_shared`` for datasets
  used by several chapters, the chapter (``03_spatial_analysis``) otherwise, or ``00_unassigned``;
- ``Tutorials/notebooks/08_remote_sensing/<section>/data``  ->  ``Data/08_remote_sensing/<section>``.

Rules (code cells of notebooks, helper ``.py`` files, tests and ``src``; markdown cells get the
string rules only):

1. string paths ``data/<dataset>`` (also after ``../`` chains or an absolute repository path)
   gain the group: ``Data/<group>/<dataset>``;
2. path joins ``"data" / "<dataset>"`` and ``"data", "<dataset>"`` likewise;
3. joins onto a variable, ``X / "<dataset>"``, gain the group (``X`` is the repository data root);
4. the repository data root itself, ``ROOT / "data"`` or ``(candidate / "data").is_dir()``, becomes
   ``"Data"``;
5. in remote sensing files, the section's own data folder (``Path("data")``, ``data/...``,
   ``NOTEBOOK_DIR / "data"``, ``Tutorials/notebooks/08_remote_sensing/<s>/data``,
   ``../<s>/data``) points to ``Data/08_remote_sensing/<section>``.

    python Tutorials/_tools/rewrite_data_paths.py --dry-run
    python Tutorials/_tools/rewrite_data_paths.py
"""

import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
NB_ROOT = REPO / "Tutorials" / "notebooks"
RS = "08_remote_sensing"
ROOT_BASES = {"ROOT", "root", "REPO", "repo", "repo_root", "REPO_ROOT", "candidate", "p", "parent", "PROJECT_ROOT",
              "project_root", "BASE", "base_dir"}
LOCAL_BASES = {"NOTEBOOK_DIR", "HERE", "NB_DIR", "nb_dir", "here", "CHAPTER", "SECTION", "Path.cwd()", "NOTEBOOK"}


def load_groups():
    groups = {}
    for r in csv.DictReader(open(REPO / "Data" / "MOVE_MANIFEST.csv")):
        o, m = Path(r["original"]), Path(r["moved_to"])
        if o.parts[0] == "data":
            groups[o.name] = m.parts[1]
    return groups


GROUPS = load_groups()
DS = "|".join(sorted((re.escape(k) for k in GROUPS), key=len, reverse=True))
SECTIONS = sorted(p.name for p in (REPO / "Data" / RS).iterdir() if p.is_dir())
SEC = "|".join(re.escape(s) for s in SECTIONS)
END = r"(?=[/\"'\s),\]}:;]|\\|$)"  # end of a path component


def section_of(f):
    try:
        parts = f.relative_to(NB_ROOT).parts
    except ValueError:
        return None
    return parts[1] if len(parts) > 2 and parts[0] == RS and parts[1] in SECTIONS else None


def rel_to(f, target):
    """Relative path from the folder of ``f`` (a notebook runs there) to ``target``."""
    return os.path.relpath(target, f.parent).replace(os.sep, "/")


class Rewriter:
    def __init__(self, f):
        self.f = f
        self.sec = section_of(f)
        self.stats = Counter()
        self.examples = defaultdict(list)

    def _sub(self, rule, pattern, repl, text, flags=0):
        def wrap(m):
            out = repl(m) if callable(repl) else m.expand(repl)
            if out != m.group(0):
                self.stats[rule] += 1
                if len(self.examples[rule]) < 3:
                    self.examples[rule].append((m.group(0), out))
            return out
        return re.sub(pattern, wrap, text, flags=flags)

    def strings(self, t):
        # rule 5 first: remote sensing chapter folders (any section), root-relative or absolute
        t = self._sub("5 Tutorials/.../<s>/data", rf"Tutorials/notebooks/{RS}/({SEC})/data{END}",
                      rf"Data/{RS}/\1", t)
        if self.sec:
            t = self._sub("5 ../<s>/data", rf"(?<![\w./-])\.\./({SEC})/data{END}",
                          lambda m: rel_to(self.f, REPO / "Data" / RS / m.group(1)), t)
        # rule 1: repository datasets in string paths
        t = self._sub("1 data/<dataset>", rf"(^|[\"'\s(=,`]|(?:\.\./)+|GeoLattice/|/)data/({DS}){END}",
                      lambda m: f"{m.group(1)}Data/{GROUPS[m.group(2)]}/{m.group(2)}", t, re.M)
        if self.sec:  # rule 5: the section's own data folder (relative to the notebook in code)
            local = rel_to(self.f, REPO / "Data" / RS / self.sec) if self.in_code else f"Data/{RS}/{self.sec}"
            t = self._sub("5 data/<local>", rf"(^|[\"'\s(=,`])data/(?!({DS}){END})([\w*.\-\[\]{{}} ]+)",
                          lambda m: f"{m.group(1)}{local}/{m.group(3)}", t, re.M)
        return t

    in_code = False

    def markdown(self, t):
        self.in_code = False
        return self.strings(t)

    def code(self, t):
        self.in_code = True
        t = self.strings(t)
        # rule 2: literal joins
        t = self._sub("2 \"data\" / \"<ds>\"", rf"([\"'])data\1(\s*/\s*)([\"'])({DS})\3",
                      lambda m: f'{m.group(1)}Data{m.group(1)}{m.group(2)}"{GROUPS[m.group(4)]}"{m.group(2)}{m.group(3)}{m.group(4)}{m.group(3)}', t)
        t = self._sub("2 \"data\", \"<ds>\"", rf"([\"'])data\1(\s*,\s*)([\"'])({DS})\3",
                      lambda m: f'{m.group(1)}Data{m.group(1)}{m.group(2)}"{GROUPS[m.group(4)]}"{m.group(2)}{m.group(3)}{m.group(4)}{m.group(3)}', t)
        # rule 3: joins onto a variable that holds the repository data root
        t = self._sub("3 X / \"<ds>\"", rf"(?<![\"'])(\b[A-Za-z_][\w.\[\]()]*\s*/\s*)([\"'])({DS})\2(?!\s*/\s*[\"']?(?:{'|'.join(set(GROUPS.values()))}))",
                      lambda m: m.group(0) if m.group(1).strip().rstrip("/").strip() in ("Data",) else
                      f'{m.group(1)}"{GROUPS[m.group(3)]}" / {m.group(2)}{m.group(3)}{m.group(2)}', t)
        # rule 5 (code): the section's own data folder
        if self.sec:
            local = rel_to(self.f, REPO / "Data" / RS / self.sec)
            t = self._sub("5 Path(\"data\")", r"Path\((['\"])data\1\)", f'Path("{local}")', t)
            t = self._sub("5 NOTEBOOK_DIR / \"data\"",
                          r"\b(" + "|".join(re.escape(b) for b in LOCAL_BASES) + r")(\s*/\s*)([\"'])data\3",
                          lambda m: f'{m.group(1)}{m.group(2)}"{local}"', t)
        # rule 4: the repository data root
        def root_only(m):
            base = re.split(r"[(\s]", m.group(1).rstrip("/ ").strip())[-1]
            ok = base in ROOT_BASES or re.fullmatch(r"Path\(__file__\)\.resolve\(\)\.parents\[\d\]", base or "")
            return f"{m.group(1)}{m.group(2)}Data{m.group(2)}" if ok else m.group(0)
        # rule 4b: a relative path to the repository data root, Path("../../../data")
        def rel_root(m):
            ok = (self.f.parent / m.group(2)).resolve() == REPO / "data"
            return f"{m.group(1)}{m.group(2)[:-4]}Data{m.group(1)}" if ok else m.group(0)
        t = self._sub("4b \"../data\"", r"([\"'])((?:\.\./)+data)\1", rel_root, t)
        t = self._sub("4 X / \"data\"", r"((?:\b[A-Za-z_][\w.]*\(|\b)(?:Path\(__file__\)\.resolve\(\)\.parents\[\d\]|[A-Za-z_]\w*)\s*/\s*)([\"'])data\2",
                      root_only, t)
        return t


def rewrite_notebook(f, dry):
    nb = json.loads(f.read_text())
    rw = Rewriter(f)
    changed = False
    for c in nb.get("cells", []):
        src = "".join(c.get("source", []))
        new = rw.code(src) if c.get("cell_type") == "code" else rw.markdown(src)
        if new != src:
            changed = True
            lines = new.split("\n")
            c["source"] = [ln + "\n" for ln in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    if changed and not dry:
        f.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    return rw


def rewrite_py(f, dry):
    t = f.read_text()
    rw = Rewriter(f)
    new = rw.code(t)
    if new != t and not dry:
        f.write_text(new)
    return rw


KEEP = {"open_cloud_native_vector_data.py", "test_open_cloud_native_vector_data.py"}  # root / "data" is an Iceberg table folder


def targets():
    for f in sorted(NB_ROOT.rglob("*")):
        if ".ipynb_checkpoints" in f.parts or f.suffix not in (".ipynb", ".py"):
            continue
        if f.suffix == ".py" and REPO / "Data" in f.parents:
            continue
        yield f
    for f in sorted((REPO / "tests").rglob("*.py")):
        if f.name not in KEEP:
            yield f
    for f in sorted((REPO / "src").rglob("*.py")):
        if f.name not in KEEP:
            yield f


def main(dry):
    total = Counter()
    examples = defaultdict(list)
    files = 0
    for f in targets():
        rw = rewrite_notebook(f, dry) if f.suffix == ".ipynb" else rewrite_py(f, dry)
        if rw.stats:
            files += 1
            total.update(rw.stats)
            for k, v in rw.examples.items():
                if len(examples[k]) < 6:
                    examples[k] += [(f.relative_to(REPO), a, b) for a, b in v][: 6 - len(examples[k])]
    print(("DRY RUN: " if dry else "") + f"{files} files, {sum(total.values())} rewrites")
    for k, v in sorted(total.items()):
        print(f"  {v:5d}  rule {k}")
        for f, a, b in examples[k][:4]:
            print(f"          {str(f)[-60:]}: {a[:70]!r} -> {b[:90]!r}")


if __name__ == "__main__":
    main("--dry-run" in sys.argv)
