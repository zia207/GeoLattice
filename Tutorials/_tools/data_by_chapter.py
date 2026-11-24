"""Data used by the tutorial notebooks, organised by chapter, sub-chapter and notebook.

Reads ``Tutorials/DATA_INVENTORY.csv`` (from ``data_inventory.py``) and writes
``Tutorials/DATA_BY_CHAPTER.md`` and ``Tutorials/DATA_BY_CHAPTER.xlsx`` (a summary sheet,
one sheet per chapter and a sheet of datasets shared between chapters).

    python Tutorials/_tools/data_by_chapter.py
"""

import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "Tutorials"
CHAPTERS = {
    "01_vector": "Vector Data", "02_raster": "Raster Data", "03_spatial_analysis": "Spatial Analysis",
    "04_spatial_statistics": "Spatial Statistics", "05_geostatistics": "Geostatistics",
    "06_spatialDB": "Spatial Databases", "07_web_cloud_gis": "Web and Cloud GIS", "08_remote_sensing": "Remote Sensing",
}


def human(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024


def size_of(p):
    """File size, or the total size of a folder dataset (.gdb, .zarr, .SAFE)."""
    if p.is_file():
        return p.stat().st_size
    if p.is_dir():
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    return 0


def dataset_of(path):
    """Dataset folder of a file: ``Data/<group>/<dataset>`` or ``Data/08_remote_sensing/<section>/<dataset>``."""
    parts = Path(path).parts
    if parts and parts[0] == "Data" and len(parts) > 2:
        depth = 4 if len(parts) > 3 and re.match(r"\d\d_\d\d_", parts[2]) else 3
        return "/".join(parts[:min(depth, len(parts) - 1)])
    return str(Path(path).parent)


def load():
    d = pd.read_csv(OUT / "DATA_INVENTORY.csv")
    d = d[d["status"].str.startswith("found") | (d["kind"] == "url")].copy()
    rel = d["notebook"].str.replace("Tutorials/notebooks/", "", regex=False)
    d["chapter"] = rel.str.split("/").str[0]
    d["section"] = rel.map(lambda r: r.split("/")[1] if r.count("/") >= 2 else "")
    d["notebook_name"] = rel.str.split("/").str[-1]
    local = d["kind"] != "url"
    d["dataset"] = ""
    d.loc[local, "dataset"] = d.loc[local, "resolved_path"].map(dataset_of)
    d["size_bytes"] = 0
    d.loc[local, "size_bytes"] = [size_of(REPO / p) for p in d.loc[local, "resolved_path"]]
    d = d[~d["reference"].str.contains(r"example\.(?:com|org)|localhost|127\.0\.0\.1", regex=True)]
    d["match"] = d["status"].map({"found": "path", "found (by file name)": "file name",
                                  "found (by glob pattern)": "glob pattern", "download URL": "download"})
    return d.drop_duplicates(["notebook", "resolved_path"])


def main():
    d = load()
    local = d[d["kind"] != "url"]
    chapters = [c for c in CHAPTERS if c in set(d["chapter"])] + sorted(set(d["chapter"]) - set(CHAPTERS))

    # chapter summary
    summ = []
    for ch in chapters:
        x, xl = d[d.chapter == ch], local[local.chapter == ch]
        uniq = xl.drop_duplicates("resolved_path")
        summ.append({"chapter": ch, "title": CHAPTERS.get(ch, ""), "notebooks with data": x.notebook.nunique(),
                     "dataset folders": uniq.dataset.nunique(), "data files": len(uniq),
                     "size": human(uniq.size_bytes.sum()), "download URLs": int((x.kind == "url").sum()),
                     "size_bytes": int(uniq.size_bytes.sum())})
    summ = pd.DataFrame(summ)

    # datasets shared between chapters
    share = (local.groupby("dataset").agg(chapters=("chapter", lambda s: ", ".join(sorted(set(s)))),
                                          n_chapters=("chapter", "nunique"), notebooks=("notebook", "nunique"),
                                          files=("resolved_path", "nunique"))
             .query("n_chapters > 1").sort_values(["n_chapters", "notebooks"], ascending=False).reset_index())

    write_markdown(d, local, chapters, summ, share)
    write_excel(d, chapters, summ, share)
    print(f"{len(chapters)} chapters | {d.notebook.nunique()} notebooks | {local.resolved_path.nunique()} data files | "
          f"{len(share)} shared datasets")


def write_markdown(d, local, chapters, summ, share):
    L = ["# Tutorial Data by Chapter", "",
         "The data each tutorial notebook reads, organised by chapter, sub-chapter and notebook. Paths are relative "
         "to the repository root. Built from `Tutorials/DATA_INVENTORY.csv` by `Tutorials/_tools/data_by_chapter.py`; "
         "the same tables are in `Tutorials/DATA_BY_CHAPTER.xlsx` (one sheet per chapter).", "",
         "## Contents", "", "- [Summary](#summary)"]
    L += [f"- [{ch} · {CHAPTERS.get(ch, '')}](#{ch.replace('_', '_')}--{CHAPTERS.get(ch, '').lower().replace(' ', '-')})"
          for ch in chapters]
    L += ["- [Datasets shared between chapters](#datasets-shared-between-chapters)", "", "## Summary", "",
          "| Chapter | Title | Notebooks with data | Dataset folders | Data files | Size | Download URLs |",
          "|---|---|---:|---:|---:|---:|---:|"]
    for r in summ.itertuples():
        L.append(f"| {r.chapter} | {r.title} | {r._3} | {r._4} | {r._5} | {r.size} | {r._7} |")
    L.append("")
    for ch in chapters:
        x = d[d.chapter == ch]
        xl = local[local.chapter == ch]
        L += [f"## {ch} · {CHAPTERS.get(ch, '')}", ""]
        ds = (xl.groupby("dataset").agg(notebooks=("notebook", "nunique"), files=("resolved_path", "nunique"),
                                        size=("size_bytes", "sum")).sort_values("notebooks", ascending=False))
        if len(ds):
            L += ["**Dataset folders used in this chapter**", "", "| Dataset folder | Notebooks | Files | Size |",
                  "|---|---:|---:|---:|"]
            L += [f"| `{k}` | {v.notebooks} | {v.files} | {human(v['size'])} |" for k, v in ds.iterrows()]
            L.append("")
        for sec in sorted(x.section.unique()):
            if sec:
                L += [f"### {sec}", ""]
            for nb in sorted(x[x.section == sec].notebook_name.unique()):
                y = x[(x.section == sec) & (x.notebook_name == nb)]
                L += [f"**{nb}**", ""]
                yl = y[y.kind != "url"]
                for folder, g in yl.groupby(yl.resolved_path.map(lambda p: str(Path(p).parent))):
                    if len(g) > 6:
                        exts = ", ".join(sorted({Path(p).suffix for p in g.resolved_path}))
                        note = " *(written by a notebook)*" if "/output" in folder else ""
                        L.append(f"- `{folder}/`: {len(g)} files ({exts}), {human(g.size_bytes.sum())}{note}")
                    else:
                        for r in g.sort_values("resolved_path").itertuples():
                            tag = "" if r.match == "path" else f" *(by {r.match})*"
                            if "/output" in r.resolved_path:
                                tag += " *(written by a notebook)*"
                            L.append(f"- `{r.resolved_path}` ({human(r.size_bytes)}){tag}")
                for u in y[y.kind == "url"].reference:
                    L.append(f"- download: <{u}>")
                L.append("")
    L += ["## Datasets shared between chapters", "", "| Dataset folder | Chapters | Notebooks | Files |",
          "|---|---|---:|---:|"]
    L += [f"| `{r.dataset}` | {r.chapters} | {r.notebooks} | {r.files} |" for r in share.itertuples()]
    (OUT / "DATA_BY_CHAPTER.md").write_text("\n".join(L) + "\n")


def write_excel(d, chapters, summ, share):
    cols = ["section", "notebook_name", "dataset", "resolved_path", "size_bytes", "match", "reference"]
    with pd.ExcelWriter(OUT / "DATA_BY_CHAPTER.xlsx", engine="openpyxl") as xw:
        summ.drop(columns="size_bytes").to_excel(xw, sheet_name="Summary", index=False)
        for ch in chapters:
            x = d[d.chapter == ch][cols].sort_values(["section", "notebook_name", "resolved_path"])
            x = x.rename(columns={"notebook_name": "notebook", "resolved_path": "path (or URL)", "size_bytes": "size (bytes)"})
            x.to_excel(xw, sheet_name=ch[:31], index=False)
        share.drop(columns="n_chapters").to_excel(xw, sheet_name="Shared datasets", index=False)
        for ws in xw.book.worksheets:  # readable column widths, frozen header
            ws.freeze_panes = "A2"
            for col in ws.columns:
                width = max(len(str(c.value or "")) for c in col[:200])
                ws.column_dimensions[col[0].column_letter].width = min(max(10, width + 2), 90)


if __name__ == "__main__":
    main()
