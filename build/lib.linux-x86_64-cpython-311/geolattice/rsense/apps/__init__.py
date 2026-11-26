"""Dashboard templates (Streamlit + leafmap) you can copy, run and customize."""

from __future__ import annotations

import shutil
from importlib import resources
from os import PathLike
from pathlib import Path

TEMPLATES = {
    "index_explorer": "Upload a GeoTIFF, map bands, compute any of 100+ spectral indices",
    "flood_mapper": "Sentinel-1 RTC flood extent for a bounding box (Planetary Computer)",
    "change_viewer": "Swipe-compare two rasters with change vector analysis",
}


def list_templates() -> dict[str, str]:
    return dict(TEMPLATES)


def create_app(template: str, path: str | PathLike = ".", *, overwrite: bool = False) -> Path:
    """Copy a template app to ``path``; run it with ``streamlit run <file>``."""
    if template not in TEMPLATES:
        raise ValueError(f"Unknown template {template!r}; choose from {list(TEMPLATES)}")
    dst = Path(path)
    dst = dst / f"{template}.py" if dst.is_dir() or not dst.suffix else dst
    if dst.exists() and not overwrite:
        raise FileExistsError(f"{dst} exists; pass overwrite=True")
    dst.parent.mkdir(parents=True, exist_ok=True)
    src = resources.files(__package__).joinpath("templates", f"{template}.py")
    with resources.as_file(src) as s:
        shutil.copy(s, dst)
    return dst


def launch(template: str, port: int = 8501) -> None:
    """Copy a template to a temp dir and launch it with Streamlit."""
    import subprocess
    import sys
    import tempfile

    p = create_app(template, tempfile.mkdtemp())
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(p), "--server.port", str(port)],
                   check=True)
