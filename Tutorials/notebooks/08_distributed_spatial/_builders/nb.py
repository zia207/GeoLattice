"""Shared helpers for the 08 distributed spatial computing notebook builders.

Each ``build_08_XX.py`` imports :class:`Notebook`, adds markdown and code cells and calls
``write()``. ``python _builders/run.py XX`` builds, executes and dumps the outputs for review;
``python _builders/sync_text.py XX`` updates only the markdown of an executed notebook. Builders
do not write the installation notes: after ``run.py`` or ``sync_text.py`` run
``Tutorials/_tools/notebook_install_notes.py`` again.
"""

from pathlib import Path

import nbformat as nbf

CHAPTER = Path(__file__).resolve().parents[1]
BANNER = "![Banner](../../../Images/Banner/09_distributed_spatial_datas.png)"

SETUP = '''
import warnings  # warning control
warnings.filterwarnings("ignore")  # hide routine library warnings
import os  # environment and CPU count
import time  # wall-clock timing
from pathlib import Path  # file paths

import matplotlib.pyplot as plt  # figures
import numpy as np  # arrays
import pandas as pd  # tables
import geopandas as gpd  # vector data
import xarray as xr  # labelled rasters

import geolattice  # GeoLattice
import geolattice.distributed as pdist  # distributed and GPU spatial computing
import dist_data as dd  # chapter helper: paths, cached downloads, timers

plt.rcParams.update({"figure.dpi": 110})  # figure resolution
pd.set_option("display.width", 140)  # wider printed tables
OUT = Path("outputs")  # results and scratch files written by this chapter
OUT.mkdir(exist_ok=True)  # create it if needed
caps = pdist.capabilities()  # engines and GPUs on this machine
print("GeoLattice", geolattice.__version__, "|", os.cpu_count(), "CPU cores |", caps["gpus"], "GPU(s)",  # versions
      f"({caps.get('gpu_memory_total', 0) / 1e9:.0f} GB)" if caps["gpus"] else "")  # GPU memory
'''


class Notebook:
    def __init__(self, filename: str):
        import os

        # sync_text.py builds into a temporary folder to keep the executed outputs
        self.path = Path(os.environ.get("NB_BUILD_TO", CHAPTER)) / filename
        self.cells = [nbf.v4.new_markdown_cell(BANNER)]

    def md(self, text: str):
        self.cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))

    def code(self, text: str):
        self.cells.append(nbf.v4.new_code_cell(text.strip("\n")))

    def setup(self, extra: str = "", note: str = ""):
        self.md("## Setup" + ("\n\n" + note.strip("\n") if note else ""))
        self.code(SETUP + ("\n" + extra.strip("\n") if extra else ""))

    def write(self):
        nb = nbf.v4.new_notebook(cells=self.cells)
        nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python",
                                      "name": "python3"},
                       "language_info": {"name": "python", "version": "3.11.16"}}
        nbf.write(nb, self.path)
        print("wrote", self.path)
