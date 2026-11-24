"""Repair markdown links between tutorial notebooks after files were renumbered or moved.

A broken link ``[text](../x/07_02_04_landsat.ipynb)`` is matched to an existing notebook by, in order:

1. the same file name somewhere else;
2. the same name without its number prefix (``landsat``), preferring the linking notebook's chapter;
3. a name that starts with the old one or that the old one starts with (``change_delta_bd`` ->
   ``change_delta_bd_landsat``), when exactly one notebook fits;
4. a close spelling (difflib ratio >= 0.85), when exactly one notebook fits.

The link is rewritten relative to the linking notebook; anchors (``#section``) are kept. Links
that match nothing, or match several notebooks equally, are listed and left alone.

    python Tutorials/_tools/fix_notebook_links.py --dry-run
    python Tutorials/_tools/fix_notebook_links.py
"""

import difflib
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
NB = REPO / "Tutorials" / "notebooks"
LINK = re.compile(r"(\[([^\]]*)\]\()([^)#\s]+\.ipynb)((?:#[^)\s]*)?\))")
# notebooks that were renamed beyond recognition (checked by hand)
RENAMED = {
    "introduction_and_criteria_data": "03_04_01_00_introduction_vector_based_suitability_analysis.ipynb",
    "atlantic_county_diabetes": "04_10_01_health_disparities.ipynb",
}


def slug(name):
    """Notebook name without its number prefix: ``07_02_04_landsat.ipynb`` -> ``landsat``."""
    return re.sub(r"^(?:\d+_)+", "", Path(name).stem).lower()


NOTEBOOKS = sorted(p for p in NB.rglob("*.ipynb") if ".ipynb_checkpoints" not in p.parts)
BY_NAME, BY_SLUG = defaultdict(list), defaultdict(list)
for p in NOTEBOOKS:
    BY_NAME[p.name].append(p)
    BY_SLUG[slug(p.name)].append(p)


def pick(cands, src):
    """One candidate, preferring the linking notebook's chapter; None when still ambiguous."""
    cands = sorted(set(cands))
    if len(cands) == 1:
        return cands[0]
    chapter = src.relative_to(NB).parts[0]
    same = [c for c in cands if c.relative_to(NB).parts[0] == chapter]
    return same[0] if len(same) == 1 else None


def match(target, src):
    name, s = Path(target).name, slug(target)
    if s in RENAMED:
        return pick(BY_NAME[RENAMED[s]], src), "renamed"
    for how, cands in (("name", BY_NAME.get(name, [])), ("slug", BY_SLUG.get(s, []))):
        if cands:
            return pick(cands, src), how
    pref = [p for k, ps in BY_SLUG.items() if len(s) >= 6 and (k.startswith(s) or s.startswith(k)) and len(k) >= 6
            for p in ps]
    if pref:
        return pick(pref, src), "prefix"
    close = difflib.get_close_matches(s, list(BY_SLUG), n=2, cutoff=0.85)
    if len(close) == 1:
        return pick(BY_SLUG[close[0]], src), "close spelling"
    return None, "no match"


def main(dry):
    fixed, unresolved, files = [], [], 0
    for f in NOTEBOOKS:
        nb = json.loads(f.read_text())
        changed = False
        for c in nb["cells"]:
            if c["cell_type"] != "markdown":
                continue
            src = "".join(c["source"])

            def repl(m):
                text, target = m.group(2), m.group(3)
                if target.startswith(("http://", "https://")) or (f.parent / target).exists():
                    return m.group(0)
                new, how = match(target, f)
                if new is None:
                    unresolved.append((f, target, how))
                    return m.group(0)
                rel = os.path.relpath(new, f.parent).replace(os.sep, "/")
                fixed.append((f, target, rel, how))
                old_name = Path(target).name
                if text.strip("`") in (old_name, target, Path(old_name).stem):  # label shows the old file name
                    text = text.replace(target, new.name).replace(old_name, new.name).replace(Path(old_name).stem, new.stem)
                return f"[{text}]({rel}{m.group(4)}"

            new = LINK.sub(repl, src)
            if new != src:
                changed = True
                lines = new.split("\n")
                c["source"] = [ln + "\n" for ln in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
        if changed:
            files += 1
            if not dry:
                f.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print(("DRY RUN: " if dry else "") + f"{len(fixed)} links fixed in {files} notebooks, {len(unresolved)} left")
    seen = set()
    for f, old, new, how in fixed:
        if (old, new) not in seen:
            seen.add((old, new))
            print(f"  [{how}] {old}  ->  {new}")
    for f, old, how in unresolved:
        print(f"  UNRESOLVED ({how}) {f.relative_to(NB)}: {old}")


if __name__ == "__main__":
    main("--dry-run" in sys.argv)
