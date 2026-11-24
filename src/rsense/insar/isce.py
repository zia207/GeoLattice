"""ISCE2 ``topsApp.py`` wrapper for Sentinel-1 TOPS interferometry (needs a conda ISCE2 install)."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Sequence
from os import PathLike
from pathlib import Path

_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<topsApp>
  <component name="topsinsar">
    <property name="Sensor name">SENTINEL1</property>
    <component name="reference">
      <property name="orbit directory">{orbit_dir}</property>
      <property name="output directory">reference</property>
      <property name="safe">{reference}</property>
    </component>
    <component name="secondary">
      <property name="orbit directory">{orbit_dir}</property>
      <property name="output directory">secondary</property>
      <property name="safe">{secondary}</property>
    </component>
    <property name="swaths">{swaths}</property>
    <property name="range looks">{range_looks}</property>
    <property name="azimuth looks">{azimuth_looks}</property>
    <property name="do unwrap">{unwrap}</property>
    <property name="unwrapper name">snaphu_mcf</property>
    <property name="do denseoffsets">False</property>
    {dem}
    {roi}
  </component>
</topsApp>
"""


def write_topsapp_xml(reference: Sequence[str], secondary: Sequence[str], workdir: str | PathLike,
                      *, orbit_dir: str = "orbits", swaths: Sequence[int] = (1, 2, 3),
                      range_looks: int = 19, azimuth_looks: int = 7, unwrap: bool = True,
                      dem: str | None = None,
                      bbox: tuple[float, float, float, float] | None = None) -> Path:
    """Write a ``topsApp.xml``. ``bbox`` is (south, north, west, east)."""
    wd = Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)
    xml = _TEMPLATE.format(
        orbit_dir=orbit_dir, reference=list(reference), secondary=list(secondary),
        swaths=list(swaths), range_looks=range_looks, azimuth_looks=azimuth_looks,
        unwrap=unwrap,
        dem=f'<property name="demFilename">{dem}</property>' if dem else "",
        roi=f'<property name="region of interest">{list(bbox)}</property>' if bbox else "",
    )
    path = wd / "topsApp.xml"
    path.write_text(xml)
    return path


def run_topsapp(xml_path: str | PathLike, *, steps: str | None = None) -> None:
    """Run ``topsApp.py`` (``steps`` e.g. ``"--start=startup --end=filter"``)."""
    exe = shutil.which("topsApp.py")
    if exe is None:
        raise RuntimeError("topsApp.py not found; install ISCE2 (conda install -c conda-forge isce2)")
    cmd = [exe, str(Path(xml_path).name)] + (steps.split() if steps else [])
    subprocess.run(cmd, cwd=Path(xml_path).parent, check=True)


def read_topsapp_outputs(workdir: str | PathLike) -> dict:
    """Load filtered interferogram, coherence and unwrapped phase from ``merged/`` via GDAL."""
    from ..core.raster import read

    m = Path(workdir) / "merged"
    out = {}
    for key, name in [("interferogram", "filt_topophase.flat.geo"), ("coherence", "phsig.cor.geo"),
                      ("unwrapped", "filt_topophase.unw.geo")]:
        f = m / name
        if f.exists():
            out[key] = read(f)
    return out
