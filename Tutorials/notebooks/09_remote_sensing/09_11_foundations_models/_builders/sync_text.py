"""Update the markdown of an executed notebook without running it again.

``python _builders/sync_text.py 04`` rebuilds notebook 04 from its builder into a temporary folder,
and when every code cell is unchanged copies the executed outputs into the new build and replaces
the notebook. Use it after editing only markdown; any change to code needs ``run.py``.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
CHAPTER = HERE.parent
num = sys.argv[1]
old_path = next(CHAPTER.glob(f"08_11_{num}_*.ipynb"))
with tempfile.TemporaryDirectory() as tmp:
    subprocess.run([sys.executable, str(HERE / f"build_08_11_{num}.py")], check=True, cwd=CHAPTER,
                   env={**os.environ, "NB_BUILD_TO": tmp}, capture_output=True)
    new = nbf.read(Path(tmp) / old_path.name, as_version=4)
old = nbf.read(old_path, as_version=4)
old_code = [c for c in old.cells if c.cell_type == "code"]
new_code = [c for c in new.cells if c.cell_type == "code"]
if [c.source for c in old_code] != [c.source for c in new_code]:
    sys.exit("code cells differ: run _builders/run.py instead")
for o, n in zip(old_code, new_code):
    n.outputs, n.execution_count = o.outputs, o.execution_count
nbf.write(new, old_path)
print("updated markdown of", old_path.name)
