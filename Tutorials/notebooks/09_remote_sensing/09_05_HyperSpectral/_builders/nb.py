"""Shared helpers for the 08_05 hyperspectral notebook builders.

Each ``build_08_05_XX.py`` imports :class:`Notebook`, adds markdown and code cells and calls
``write()``. ``python _builders/run.py XX`` builds, executes and dumps the outputs for review.
"""

from pathlib import Path

import nbformat as nbf

CHAPTER = Path(__file__).resolve().parents[1]
BANNER = "![Banner](../../../../Images/Banner/07_remote_sensing/09_05_banner_rs_hyperspect.png)"

SETUP = '''
import warnings  # warning control
warnings.filterwarnings("ignore")  # hide routine library warnings

from pathlib import Path  # file paths

import numpy as np  # arrays
import pandas as pd  # tables
import matplotlib.pyplot as plt  # figures

import geolattice  # GeoLattice
from geolattice.rsense import hyperspectral as hs  # hyperspectral cubes, readers and tools

plt.rcParams.update({"figure.dpi": 110})  # figure resolution
DATA = Path("../../../../Data/08_remote_sensing/08_05_HyperSpectral")  # data folder next to this notebook
TANAGER = DATA / "Tanger"  # Planet Tanager sample scenes
OUT = Path("outputs")  # results written by this notebook
OUT.mkdir(exist_ok=True)  # create it if needed


def tanager_file(category, product="sr"):
    """Path of a Tanager product (sr or radiance) for a scene category folder."""
    return next((TANAGER / category).glob(f"*/*_ortho_{product}_hdf5.h5"))


print("GeoLattice", geolattice.__version__)  # library version used for this notebook
'''


class Notebook:
    def __init__(self, filename: str):
        self.path = CHAPTER / filename
        self.cells = [nbf.v4.new_markdown_cell(BANNER)]

    def md(self, text: str):
        self.cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))

    def code(self, text: str):
        self.cells.append(nbf.v4.new_code_cell(text.strip("\n")))

    def setup(self, extra: str = ""):
        self.md("## Setup")
        self.code(SETUP + ("\n" + extra.strip("\n") if extra else ""))

    def write(self):
        nb = nbf.v4.new_notebook(cells=self.cells)
        nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python",
                                      "name": "python3"},
                       "language_info": {"name": "python", "version": "3.11.16"}}
        nbf.write(nb, self.path)
        print("wrote", self.path)
