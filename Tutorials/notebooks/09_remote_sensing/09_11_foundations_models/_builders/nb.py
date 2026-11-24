"""Shared helpers for the 08_11 foundation model notebook builders.

Each ``build_08_11_XX.py`` imports :class:`Notebook`, adds markdown and code cells and calls
``write()``. ``python _builders/run.py XX`` builds, executes and dumps the outputs for review.
Notebooks 01 and 02 were moved here from chapter 08_09 and have no builder. Builders do not write
the installation notes: after ``run.py`` or ``sync_text.py`` run
``Tutorials/_tools/notebook_install_notes.py`` again.
"""

from pathlib import Path

import nbformat as nbf

CHAPTER = Path(__file__).resolve().parents[1]
BANNER = "![Banner](../../../../Images/Banner/07_remote_sensing/09_11_banner_rs_foundation_models.png)"

SETUP = '''
import os  # environment variables
os.environ.setdefault("USE_TF", "0")  # keep transformers from loading TensorFlow
import time  # timing
import warnings  # warning control
warnings.filterwarnings("ignore")  # hide routine library warnings
from pathlib import Path  # file paths

import matplotlib.pyplot as plt  # figures
import numpy as np  # arrays
import pandas as pd  # tables
import rioxarray  # noqa: F401  (registers the .rio accessor)
import torch  # PyTorch: every model runs on it
import xarray as xr  # labelled rasters

import geolattice  # GeoLattice
import geolattice.rsense.foundation_models as fm  # catalog, loaders, embeddings and probes

plt.rcParams.update({"figure.dpi": 110})  # figure resolution
DATA = Path("../../../../Data/08_remote_sensing")  # remote sensing data folder
IC = DATA / "09_09_image_classification"  # data shared with the image classification chapter
CAP = DATA / "08_11_Capstones"  # data shared with the capstones
OUT = Path("outputs")  # results written by this notebook
OUT.mkdir(exist_ok=True)  # create it if needed
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # run models on the GPU when there is one
SEED = 0  # one seed for every random step
np.random.seed(SEED)  # NumPy
torch.manual_seed(SEED)  # PyTorch

print("GeoLattice", geolattice.__version__, "| PyTorch", torch.__version__, "| device", DEVICE)  # versions used
if DEVICE == "cuda":  # name the GPU
    print("GPU:", torch.cuda.get_device_name(0))  # GPU model
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
