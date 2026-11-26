"""Shared helpers for the 09_13 satellite soil moisture notebook builders.

Each ``build_09_13_XX.py`` imports :class:`Notebook`, adds markdown and code cells and calls
``write()``. ``python _builders/run.py XX`` builds, executes and dumps the outputs for review;
``python _builders/sync_text.py XX`` updates only the markdown of an executed notebook. Builders
do not write the installation notes: after ``run.py`` or ``sync_text.py`` run
``Tutorials/_tools/notebook_install_notes.py`` again.
"""

from pathlib import Path

import nbformat as nbf

CHAPTER = Path(__file__).resolve().parents[1]
# the banner is Images/Banner/07_remote_sensing/09_13_banner_rs_soil_moisture.png, copied into the site's images folder
BANNER = "![Banner](../images/09_13_banner_rs_soil_moisture.png)"

SETUP = '''
import warnings  # warning control
warnings.filterwarnings("ignore")  # hide routine library warnings
from pathlib import Path  # file paths

import matplotlib.pyplot as plt  # figures
import numpy as np  # arrays
import pandas as pd  # tables
import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr  # labelled rasters

import geolattice  # GeoLattice
import geolattice.rsense.soil_moisture as sm  # soil moisture physics, products, retrieval and validation
import sm_data as sd  # chapter helper: subsets, downloads and caches the data

plt.rcParams.update({"figure.dpi": 110})  # figure resolution
OUT = Path("outputs")  # results written by this chapter
OUT.mkdir(exist_ok=True)  # create it if needed
print("GeoLattice", geolattice.__version__, "| data folder", sd.DATA.relative_to(sd.DATA.parents[3]))  # versions and paths
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
