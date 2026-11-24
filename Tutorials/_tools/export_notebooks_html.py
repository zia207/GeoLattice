"""Run the notebooks of one tutorial folder and save them as self-contained HTML pages.

Each notebook is executed in place (in name order, from its own folder, so outputs written by one
notebook are there for the next), then exported to ``docs/notebooks/<same relative folder>/`` with:

- all code output and figures;
- every image embedded, both markdown images and ``<img src="...">`` tags (banners, diagrams);
- links to other notebooks pointing to their ``.html`` page when that page exists.

    python Tutorials/_tools/export_notebooks_html.py Tutorials/notebooks/02_raster
    python Tutorials/_tools/export_notebooks_html.py Tutorials/notebooks/02_raster --no-exec   # export only
    python Tutorials/_tools/export_notebooks_html.py path/to/one.ipynb path/to/other.ipynb   # single notebooks
"""

import base64
import json
import mimetypes
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
NB_ROOT = REPO / "Tutorials" / "notebooks"
DOCS = REPO / "docs" / "notebooks"
PY = sys.executable
IMG = re.compile(r'(<img\b[^>]*?\bsrc=")([^"]+)(")')
LINK = re.compile(r'(href=")([^"#:]+)\.ipynb([#"])')


def execute(nb, timeout=5400):
    r = subprocess.run([PY, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
                        f"--ExecutePreprocessor.timeout={timeout}", "--ExecutePreprocessor.kernel_name=python3",
                        nb.name], cwd=nb.parent, capture_output=True, text=True)
    if r.returncode:
        err = [ln for ln in r.stderr.splitlines() if re.search(r"Error|error:", ln) and "MissingID" not in ln]
        return "\n".join(err[-4:]) or r.stderr[-800:]
    return ""


def export(nb, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([PY, "-m", "jupyter", "nbconvert", "--to", "html", "--HTMLExporter.embed_images=True",
                    "--output-dir", str(out_dir), nb.name], cwd=nb.parent, capture_output=True, check=True)
    page = out_dir / f"{nb.stem}.html"
    html, missing = page.read_text(), []

    def embed(m):
        src = m.group(2)
        if src.startswith(("data:", "http://", "https://")):
            return m.group(0)
        path = (nb.parent / src).resolve()
        if not path.is_file():
            missing.append(src)
            return m.group(0)
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        return f"{m.group(1)}data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}{m.group(3)}"

    def relink(m):
        target = (nb.parent / f"{m.group(2)}.ipynb").resolve()
        try:
            html_page = DOCS / target.relative_to(NB_ROOT).with_suffix(".html")
        except ValueError:
            return m.group(0)
        rel = os.path.relpath(html_page, out_dir).replace(os.sep, "/")
        return f"{m.group(1)}{rel}{m.group(3)}"

    html = LINK.sub(relink, IMG.sub(embed, html))
    page.write_text(html)
    return page, missing


def summary(nb):
    cells = [c for c in json.loads(nb.read_text())["cells"] if c["cell_type"] == "code"]
    ran = sum(1 for c in cells if c.get("execution_count"))
    errors = sum(1 for c in cells for o in c.get("outputs", []) if o.get("output_type") == "error")
    figs = sum(1 for c in cells for o in c.get("outputs", []) if "image/png" in o.get("data", {}))
    return len(cells), ran, errors, figs


def main(folders, run=True):
    for folder in folders:
        run_folder(folder, run)


def run_folder(folder, run=True):
    folder = Path(folder).resolve()
    if folder.suffix == ".ipynb":  # a single notebook
        notebooks = [folder]
    else:
        notebooks = sorted(p for p in folder.rglob("*.ipynb") if ".ipynb_checkpoints" not in p.parts
                           and not any(part.startswith("_") for part in p.relative_to(folder).parts))
    print(f"{'notebook':48s} {'cells':>5s} {'ran':>4s} {'err':>3s} {'figs':>4s} {'html MB':>7s}  time  notes")
    for nb in notebooks:
        t0 = time.time()
        problem = execute(nb) if run else ""
        page, missing = export(nb, DOCS / nb.parent.relative_to(NB_ROOT))
        n, ran, err, figs = summary(nb)
        notes = ("FAILED: " + problem.replace("\n", " | ")[:300]) if problem else ""
        if missing:
            notes += f" missing images: {missing}"
        print(f"{nb.name[:48]:48s} {n:5d} {ran:4d} {err:3d} {figs:4d} {page.stat().st_size / 1e6:7.1f} "
              f"{time.time() - t0:5.0f}s  {notes}", flush=True)


if __name__ == "__main__":
    main([a for a in sys.argv[1:] if not a.startswith("--")], run="--no-exec" not in sys.argv)
