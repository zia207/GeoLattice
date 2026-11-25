"""Render the Geostatistics quick reference (two columns of functions, tutorials below) as one PNG.

Reuses the layout, CSS and logos of make_cheatsheet_spatial_analysis.py.

    python Images/Banner/_src/make_cheatsheet_geostatistics.py
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_cheatsheet_spatial_analysis as sa  # noqa: E402

base = sa.base
HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "05_geostat" / "05_00_geostatistics_quick_reference.png"
COPY = HERE.parents[2] / "Tutorials" / "notebooks" / "05_geostatistics" / "05_geostatistics_quick_reference.png"

# Each group: (heading, tag shown on the right of the heading, rows). Every name is gstat.<name>.
ESTIMATION = [
    ("Variograms", "gstat", [
        ("Variogram", "Empirical variogram and fitted model"),
        ("spherical · exponential · gaussian · matern · stable", "Variogram models"),
        ("VariogramCloud · compute_variogram_cloud", "Variogram cloud diagnostics"),
        ("fit_nested_variogram · transform_aniso", "Nested structures, geometric anisotropy"),
        ("JaxVariogram", "Variograms with JAX on CPU or GPU"),
    ]),
    ("Kriging", "gstat", [
        ("SimpleKriging · OrdinaryKriging", "Known and unknown mean"),
        ("UniversalKriging", "Polynomial drift"),
        ("Cokriging · CrossVariogram · fit_lmc", "Cokriging, linear model of coregionalization"),
        ("MultivariateCokriging · ColocatedCokriging", "Many and colocated secondaries"),
        ("kt3d · kt3d_grid", "GSLIB style 3-D kriging"),
        ("JaxOrdinaryKriging · JaxUniversalKriging", "Kriging with JAX on CPU or GPU"),
    ]),
    ("Regression kriging", "gstat + extras", [
        ("RegressionKriging · GAMRegressor", "scikit-learn and GAM trends, kriged residuals"),
        ("RegressionKrigingH2O", "H2O AutoML, GBM, DRF, deep learning"),
        ("DeepRegressionKriging · DeepRegressionKrigingTF", "PyTorch or TensorFlow MLP trends"),
        ("GNNRegressionKriging · KCN · IGNNK", "Graph neural network kriging"),
    ]),
    ("Specialized kriging", "gstat", [
        ("IndicatorKriging · fit_indicator_variogram", "Probability of exceeding thresholds"),
        ("compute_etype_from_probabilities", "E-type estimate from probabilities"),
        ("DisjunctiveKriging", "Nonlinear estimator via Hermite polynomials"),
        ("SoftKriging · markov_bayes_calibrate", "Hard plus soft data, Markov-Bayes"),
        ("FactorialKriging", "Separate nested structures, denoise"),
    ]),
    ("Space-time kriging", "gstat", [
        ("empirical_st_variogram · fit_st_variogram · vgm_st", "Space-time variograms"),
        ("STKriging · krige_st", "Space-time ordinary kriging"),
        ("CNNLSTMRegressionKriging · ConvLSTM", "Raster space-time residual kriging"),
        ("STGNNRegressionKriging · TGCN · DCRNN", "Recurrent graph networks"),
        ("STGTNRegressionKriging · STTN · GMAN", "Graph transformer networks"),
    ]),
]

UNCERTAINTY = [
    ("Poisson kriging", "gstat", [
        ("PoissonKriging", "Centroid based kriging of rates"),
        ("AreaPoissonKriging · PointSupport", "Area-to-area and area-to-point"),
        ("poisson_cokrige · fit_poisson_lmc", "Multivariate Poisson cokriging"),
        ("STPoissonKriging", "Space-time rate panels"),
    ]),
    ("Gaussian and indicator simulation", "gstat", [
        ("sgsim · sgsim_gpu", "Sequential Gaussian simulation, GPU option"),
        ("sisim", "Sequential indicator simulation"),
        ("sgcosim · sicosim", "Gaussian and indicator co-simulation"),
        ("dssim_direct", "Direct sequential simulation"),
        ("lusim · tbsim · fftsim", "LU, turning bands and FFT simulation"),
        ("sasim", "Simulated annealing"),
    ]),
    ("Categorical and multiple-point", "gstat", [
        ("pgsim · PluriGaussianSimulator", "Plurigaussian facies simulation"),
        ("dssim · DirectSampling", "Direct Sampling from a training image"),
        ("snesim · SNESIM · tau_model", "SNESIM, combine soft probabilities"),
        ("channel_training_image · pattern_histogram", "Training images and pattern checks"),
    ]),
    ("Bayesian kriging", "gstat", [
        ("EmpiricalBayesianKriging", "Ensemble over variogram uncertainty"),
        ("BayesianKriging · bayesian_kriging", "Priors on range and nugget"),
        ("ConjugateBayesianKriging", "Closed form, known correlation"),
        ("BayesianTransformedKriging · BayesianGLMKriging", "Transformed Gaussian, spatial GLMM"),
        ("correlation_function", "Correlation from a variogram model"),
    ]),
    ("Data, transforms and GSLIB", "gstat", [
        ("from_geodataframe · make_prediction_grid", "Coordinates in, prediction grid out"),
        ("read_gslib · write_gslib · read_gslib_grid", "GSLIB and Geo-EAS files"),
        ("nscore_forward · nscore_back", "Normal score transform"),
        ("transform_boxcox · transform_log", "Box-Cox and log transforms"),
        ("declus_cell", "Cell declustering weights"),
    ]),
    ("Validation", "gstat", [
        ("loo_cv · kfold_cv", "Leave-one-out and k-fold cross-validation"),
        ("LOOCV_RK · KFoldCV_RK · IndicatorKrigingCV", "Cross-validation for each model family"),
    ]),
]

TOPICS = [
    ("05_01", "Foundation", "2", "Getting started, variogram modeling"),
    ("05_02", "Kriging", "6", "Simple, ordinary, universal, cokriging, JAX"),
    ("05_03", "Regression Kriging", "6", "scikit-learn, H2O, PyTorch, TensorFlow, GNN"),
    ("05_04", "Specialized Kriging", "6", "Indicator, 3-D, soft, disjunctive, factorial"),
    ("05_05", "Space-time Kriging", "5", "ST kriging, ConvLSTM, STGNN, STGTN"),
    ("05_06", "Poisson Kriging", "4", "Area-to-area, area-to-point, cokriging, ST"),
    ("05_07", "Geostatistical Simulation", "10", "SGS, SIS, DSS, SA, PGS, MPS, SNESIM"),
    ("05_08", "Bayesian Kriging", "8", "EBK, Bayesian, GLM, transformed, conjugate"),
    ("05_09", "Capstone Projects", "8", "Arsenic, cancer, diabetes, groundwater"),
]

IMPORTS = ("import geolattice.geostatistics as gstat<br>"
           "from geolattice.geostatistics import Variogram, OrdinaryKriging, sgsim")

tut = "".join(f'<div class="t"><b>{n}</b><div>{ti}<small><em>{k} notebooks</em> · {d}</small></div></div>'
              for n, ti, k, d in TOPICS)
total = sum(int(k) for _, _, k, _ in TOPICS)

HTML = f"""<!doctype html><html><head><meta charset="utf-8"><style>{sa.CSS}</style></head><body><div id="page">
<div class="bar"></div>
<div class="upatta"><img src="data:image/png;base64,{base.UPATTA}"></div>
<div class="kick"><div class="logo"><img src="data:image/png;base64,{base.LOGO}"></div>
<div class="k">QUICK REFERENCE · DR. ZIA U. AHMED</div></div>
<h1>Geostatistics <span>with GeoLattice</span></h1>
<div class="sub">Key functions of <b>geolattice.geostatistics</b>, grouped by task, with the tutorials that teach them.</div>
<div class="imps">{IMPORTS}</div>
<div class="cols">
{sa.column("PART A · ESTIMATION", "Variograms &amp; the kriging family", ESTIMATION)}
{sa.column("PART B · AREAS, SIMULATION &amp; UNCERTAINTY", "Poisson, simulation, Bayesian &amp; tools", UNCERTAINTY)}
</div>
<div class="tutall"><h3>Tutorials · Tutorials/notebooks/05_geostatistics · {total} notebooks</h3><div class="tgrid">{tut}</div></div>
<div class="foot"><span>GEOSTATISTICS WITH GEOLATTICE · UPATTA ANALYTICS</span>
<span>call any name as gstat.name(...) · H2O and TensorFlow models need their extras</span></div>
</div></body></html>"""

if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "sheet.html"
        src.write_text(HTML)
        raw = Path(tmp) / "raw.png"
        subprocess.run(["google-chrome", "--headless=new", "--disable-gpu", "--hide-scrollbars",
                        "--force-device-scale-factor=2", "--window-size=1400,3200",
                        "--virtual-time-budget=4000", f"--screenshot={raw}", src.as_uri()],
                       check=True, capture_output=True)
        im = Image.open(raw).convert("RGB")
        # crop below the content: the yellow side bar spans exactly the page
        bar = np.asarray(im).astype(int)[:, 6]
        yellow = (bar[:, 0] > 200) & (bar[:, 1] > 170) & (bar[:, 2] < 80)
        last = int(np.flatnonzero(yellow).max()) + 1
        if last >= im.height - 2:
            print("warning: page may be taller than the window; raise --window-size height")
        im.crop((0, 0, im.width, last)).save(OUT)
    shutil.copyfile(OUT, COPY)
    print("wrote", OUT, Image.open(OUT).size, "and", COPY)
