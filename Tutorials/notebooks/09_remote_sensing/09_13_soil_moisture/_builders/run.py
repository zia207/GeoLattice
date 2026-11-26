"""Build, execute and dump one notebook: ``python _builders/run.py 03 [--no-exec]``.

Cells run with ``--allow-errors`` so every failing cell shows up in one pass; a notebook is ready
only when the dump has no ``ERROR`` lines.

Outputs (text and PNG figures) are written to ``$NB_DUMP`` (default ``<tempdir>/nbdump``)/<nn>/.
"""

import base64
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHAPTER = HERE.parent
num = sys.argv[1]
builder = HERE / f"build_09_13_{num}.py"
subprocess.run([sys.executable, str(builder)], check=True, cwd=CHAPTER)
nb_path = next(CHAPTER.glob(f"09_13_{num}_*.ipynb"))
if "--no-exec" not in sys.argv:
    r = subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook",
                        "--execute", "--inplace", "--allow-errors", "--ExecutePreprocessor.timeout=3600",
                        "--ExecutePreprocessor.kernel_name=python3", str(nb_path)],
                       cwd=CHAPTER, capture_output=True, text=True)
    print(r.stderr[-3000:])
dump = Path(os.environ.get("NB_DUMP", Path(tempfile.gettempdir()) / "nbdump")) / num
dump.mkdir(parents=True, exist_ok=True)
for f in dump.glob("*"):
    f.unlink()
nb = json.loads(nb_path.read_text())
n = 0
lines = []
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    for o in c.get("outputs", []):
        d = o.get("data", {})
        if o.get("output_type") == "error":
            lines.append(f"[{i}] ERROR {o['ename']}: {o['evalue']}")
            lines.append("\n".join(o.get("traceback", []))[-3000:])
        elif "image/png" in d:
            name = f"cell{i:02d}_{n}.png"
            (dump / name).write_bytes(base64.b64decode(d["image/png"]))
            lines.append(f"[{i}] IMG {name}")
            n += 1
        elif "text/plain" in d and not any(k.startswith("application/vnd") for k in d):
            lines.append(f"[{i}] OUT " + "".join(d["text/plain"])[:2500])
        elif "text" in o:
            lines.append(f"[{i}] STD " + "".join(o["text"])[:2500])
        else:
            lines.append(f"[{i}] OTHER {list(d)}")
(dump / "outputs.txt").write_text("\n".join(lines))
print("\n".join(lines))
print("size MB", round(nb_path.stat().st_size / 2**20, 1))
