"""Render the Spatial Causal Inference quick reference (two columns, tutorials below) as one PNG.

Reuses the layout, CSS and logos of make_cheatsheet_spatial_analysis.py. Every name is checked
against geolattice.spatial_analysis.spatialcausal (imported as sc).

    python Images/Banner/_src/make_cheatsheet_spatial_causal.py
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
ROOT = HERE.parents[2]
OUT = HERE.parent / "03_spatial_analysis" / "03_07_spatial_causal_quick_reference.png"
COPY = (ROOT / "Tutorials" / "notebooks" / "03_spatial_analysis" / "03_07_spatial_causal"
        / "03_07_spatial_causal_quick_reference.png")

# Each group: (heading, tag, rows). The tag is sc or a submodule of it.
IDENTIFY = [
    ("Data and results", "sc", [
        ("SpatialCausalData · SpatialPanel", "Units with locations, or unit by time panels"),
        ("SpatialCausalResult · EffectEstimate", "Effects with standard errors and intervals"),
    ]),
    ("Simulated data", "sc.datasets", [
        ("simulate_spatial_confounding · simulate_interference", "Confounding and spillover scenarios"),
        ("simulate_geographic_rdd · simulate_spatial_did", "Boundary and panel designs"),
        ("simulate_synthetic_control · simulate_spatial_dag", "Synthetic control, causal graphs"),
    ]),
    ("Spatial confounding", "sc", [
        ("diagnose_spatial_confounding", "Moran's I on treatment, outcome, residuals"),
        ("moran_eigenvectors · rbf_basis · SpatialBasis", "Spatial smooths as covariates"),
        ("SpatialPlus", "spatial+ adjustment"),
        ("compare_adjustments", "OLS, smooth, spatial+ and DML side by side"),
    ]),
    ("Interference and spillovers", "sc", [
        ("kernel_weights · DistanceDecayExposure", "Distance-decay exposure to treated units"),
        ("ExposureMapping · BipartiteExposure", "Exposure conditions, treatments elsewhere"),
        ("BernoulliDesign · CompleteDesign · ClusterDesign", "Assignment mechanisms"),
        ("exposure_probabilities", "Probability of each exposure condition"),
        ("InterferenceEstimator", "Direct and spillover effects"),
    ]),
    ("Double machine learning", "sc", [
        ("spatial_features", "Add coordinates and spatial lags to X"),
        ("SpatialDML · SpatialDRLearner", "Partially linear DML, doubly robust CATE"),
        ("SpatialBlockKFold · make_folds", "Spatial cross-fitting folds"),
        ("PyCausalMLAdapter", "Any PyCausalML estimator, spatial features"),
    ]),
    ("Inference under dependence", "sc", [
        ("standard_error · conley_variance · suggest_cutoff", "Conley (spatial HAC) errors"),
        ("spatial_block_bootstrap · block_permutation_test", "Block bootstrap and randomisation tests"),
    ]),
]

DESIGNS = [
    ("Spatially varying effects", "sc", [
        ("GWDML", "Geographically weighted DML"),
        ("GWCausalForest", "Geographically weighted causal forest"),
    ]),
    ("Geographic regression discontinuity", "sc", [
        ("GeographicRDD · ik_bandwidth", "Sharp RDD at a border, optimal bandwidth"),
    ]),
    ("Panels and synthetic control", "sc", [
        ("SpatialDiD", "Two-way fixed effects with distance rings"),
        ("StaggeredSpatialDiD", "Staggered adoption with spillovers"),
        ("SpatialSyntheticControl", "Synthetic control with a donor buffer"),
    ]),
    ("Bayesian models", "sc", [
        ("BayesianSpatialCausal", "Treatment and spillovers with ICAR or BYM"),
    ]),
    ("Designing spatial experiments", "sc", [
        ("ClusterRandomizedDesign · SaturationDesign", "Cluster and two-stage saturation designs"),
        ("balanced_allocation · rerandomize", "GRTS balanced allocation, rerandomisation"),
        ("design_diagnostics", "Balance and spatial structure"),
        ("power_analysis · power_curve", "Simulation based power"),
    ]),
    ("Causal discovery", "sc", [
        ("SpatialGranger", "Granger causality with spatial lags"),
        ("GCCM", "Geographical convergent cross mapping"),
        ("SpatialNOTEARS · notears_linear", "Learn causal graphs with spillover edges"),
    ]),
    ("Plots", "sc.viz", [
        ("plot_effects · plot_effect_map · plot_local_effects", "Forest plots and effect maps"),
        ("plot_rdd · plot_event_study · plot_synthetic_control", "Design specific plots"),
        ("plot_exposure_conditions · plot_cutoff_sensitivity", "Exposure maps, Conley cutoff check"),
    ]),
]

NOTEBOOK_LIST = [
    ("03_07_00", "Spatial Causal Inference", "Data and causal setup"),
    ("03_07_01", "Spatial Confounding", "Diagnose, spatial+, DML"),
    ("03_07_02", "Interference and Spillovers", "Exposure mappings, spillover effects"),
    ("03_07_03", "Spatially Varying Effects", "GWDML, causal forests"),
    ("03_07_04", "Geographic RDD", "Effects at a boundary"),
    ("03_07_05", "Panel Designs", "DiD, staggered adoption, synthetic control"),
    ("03_07_06", "Bayesian Spatial Causal Models", "ICAR and BYM"),
    ("03_07_07", "Designing Spatial Experiments", "Allocation, balance, power"),
    ("06_03_08", "Spatial Causal Discovery", "Granger, GCCM, NOTEARS"),
]

IMPORTS = ('import geolattice.spatial_analysis.spatialcausal as sc<br>'
           'data = sc.datasets.simulate_spatial_confounding(n=400, random_state=0)<br>'
           'print(sc.SpatialDML().fit(data).summary())')

CSS = sa.CSS + """
.t { grid-template-columns: 92px 1fr; }
"""

tut = "".join(f'<div class="t"><b>{n}</b><div>{ti}<small>{d}</small></div></div>' for n, ti, d in NOTEBOOK_LIST)

HTML = f"""<!doctype html><html><head><meta charset="utf-8"><style>{CSS}</style></head><body><div id="page">
<div class="bar"></div>
<div class="upatta"><img src="data:image/png;base64,{base.UPATTA}"></div>
<div class="kick"><div class="logo"><img src="data:image/png;base64,{base.LOGO}"></div>
<div class="k">QUICK REFERENCE · DR. ZIA U. AHMED</div></div>
<h1>Spatial Causal Inference <span>with GeoLattice</span></h1>
<div class="sub">Key functions of <b>geolattice.spatial_analysis.spatialcausal</b>, grouped by task, with the tutorials that teach them.</div>
<div class="imps">{IMPORTS}</div>
<div class="cols">
{sa.column("PART A · IDENTIFICATION", "Confounding, spillovers &amp; inference", IDENTIFY)}
{sa.column("PART B · DESIGNS &amp; DISCOVERY", "Local effects, quasi-experiments &amp; graphs", DESIGNS)}
</div>
<div class="tutall"><h3>Tutorials · Tutorials/notebooks/03_spatial_analysis/03_07_spatial_causal · {len(NOTEBOOK_LIST)} notebooks</h3><div class="tgrid">{tut}</div></div>
<div class="foot"><span>SPATIAL CAUSAL INFERENCE WITH GEOLATTICE · UPATTA ANALYTICS</span>
<span>estimators follow est.fit(data) then est.summary() · se="conley", cutoff=d gives spatial errors</span></div>
</div></body></html>"""


def check_names():
    """Return the names on the card that the spatialcausal package does not provide."""
    import importlib

    bad = []
    for groups in (IDENTIFY, DESIGNS):
        for _, tag, rows in groups:
            path = "geolattice.spatial_analysis.spatialcausal" + tag[2:]
            mod = importlib.import_module(path)
            bad += [f"{tag}: {n}" for names, _ in rows for n in names.split(" · ") if not hasattr(mod, n)]
    return bad


if __name__ == "__main__":
    missing = check_names()
    if missing:
        raise SystemExit(f"names not found: {missing}")
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "sheet.html"
        src.write_text(HTML)
        raw = Path(tmp) / "raw.png"
        subprocess.run(["google-chrome", "--headless=new", "--disable-gpu", "--hide-scrollbars",
                        "--force-device-scale-factor=2", "--window-size=1400,3400",
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
