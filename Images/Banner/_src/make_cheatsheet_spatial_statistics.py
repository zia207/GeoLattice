"""Render the Spatial Statistics quick reference (two columns of functions, tutorials below) as one PNG.

Reuses the layout, CSS and logos of make_cheatsheet_spatial_analysis.py.

    python Images/Banner/_src/make_cheatsheet_spatial_statistics.py
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
OUT = HERE.parent / "04_spatial_stat" / "04_00_spatial_statistics_quick_reference.png"
COPY = HERE.parents[2] / "Tutorials" / "notebooks" / "04_spatial_statistics" / "04_spatial_statistics_quick_reference.png"

# Each group: (heading, subpackage tag shown on the right of the heading, rows)
PATTERNS = [
    ("Spatial weights", "weights", [
        ("Queen · Rook", "Contiguity neighbours"),
        ("KNN · DistanceBand · Kernel", "Distance based neighbours"),
        ("Delaunay · Gabriel · RelativeNeighborhood", "Graph based neighbours"),
        ("higher_order · W", "Higher order lags, custom weights"),
    ]),
    ("Point patterns", "pointpattern", [
        ("PointPattern · Window", "Points and their study area"),
        ("quadrat_test · clark_evans", "Tests against complete randomness"),
        ("kde · bandwidth_scott · bandwidth_cv", "Intensity surfaces"),
        ("ripley_k · ripley_l · pair_correlation", "Second order structure"),
        ("g_function · f_function · j_function", "Nearest neighbour functions"),
        ("cross_k · k_inhom · relative_risk", "Two types, varying intensity, case control"),
        ("simulate_csr · simulate_thomas · envelope", "Simulation envelopes"),
    ]),
    ("Spatial autocorrelation", "esda", [
        ("Moran · Geary · GeneralG", "Global autocorrelation"),
        ("Moran_Local · Geary_Local · G_Local", "Local indicators (LISA)"),
        ("MoranBV · Moran_Local_BV", "Bivariate Moran"),
        ("Join_Counts", "Binary and categorical data"),
        ("conditional_permutation · fdr_bh", "Permutation inference, FDR"),
    ]),
    ("Maps", "viz", [
        ("choropleth · moran_scatter · lisa_cluster_map", "Maps, Moran plot, LISA clusters"),
    ]),
    ("Spatial clusters", "cluster", [
        ("getis_ord_gi · local_moran_clusters", "Hot spots and LISA clusters"),
        ("spatial_scan", "Kulldorff scan statistic"),
        ("dbscan_clusters · hdbscan_clusters · k_distance", "Density based point clusters"),
        ("skater · agglomerative · evaluate_regions", "Regionalisation"),
        ("partition_agreement · jaccard · flag_stability", "Compare and validate clusters"),
        ("adjust_pvalues", "FDR or Bonferroni adjustment"),
    ]),
    ("Disease mapping", "bayes", [
        ("expected_counts · smr · smr_confidence_interval", "Standardised mortality ratios"),
        ("excess_pvalue · funnel_limits", "Excess risk, funnel plots"),
        ("eb_gamma_poisson · eb_james_stein · eb_local", "Empirical Bayes smoothing"),
        ("BYM · icar_structure", "Bayesian BYM model"),
        ("rhat · ess", "MCMC convergence"),
        ("shrinkage_summary · evaluate_smoothing", "What smoothing did"),
    ]),
]

MODELS = [
    ("Spatial regression", "regression", [
        ("OLS · moran_residuals", "Baseline model, residual autocorrelation"),
        ("jarque_bera · breusch_pagan · condition_number", "Normality, heteroskedasticity, collinearity"),
        ("lm_tests", "LM and robust LM tests for lag or error"),
        ("SpatialLag · SpatialError", "Spatial lag and error models"),
        ("SpatialDurbin · SLX", "Durbin and lagged X models"),
        ("lr_test · compare_models", "Compare and report models"),
    ]),
    ("Geographically weighted models", "gwmodel", [
        ("GWSS", "Local summary statistics"),
        ("GWR · MGWR", "Local regression, one or many bandwidths"),
        ("GWGLM · GTWR", "Count and binary outcomes, space and time"),
        ("GWRandomForest · GWXGBoost · GWLightGBM", "GW machine learning"),
        ("GNNWR · GTNNWR", "Neural network weighted regression"),
        ("GWPCA · NetworkGWR", "Local PCA, network distances"),
        ("set_compute_options", "CPU or GPU, threads"),
    ]),
    ("Spatial interpolation", "interpolate", [
        ("NearestNeighbor · Thiessen · TIN", "Nearest and triangulated surfaces"),
        ("IDW · RBF · Spline · TrendSurface", "Deterministic surfaces"),
        ("empirical_variogram · fit_variogram_auto", "Variograms"),
        ("OrdinaryKriging · UniversalKriging · RegressionKriging", "Kriging with covariates"),
        ("Cokriging · IndicatorKriging", "Multivariate and indicator kriging"),
        ("areal_weighting · dasymetric", "Areal interpolation"),
        ("cross_validate · compare_methods", "Validate and compare methods"),
    ]),
    ("Spatial sampling", "sampling", [
        ("simple_random · systematic · stratified · two_stage", "Probability designs"),
        ("grts · space_filling · clhs", "Spatially balanced designs"),
        ("neyman_allocation · sample_size_mean", "Allocation and sample size"),
        ("estimate_mean · local_mean_variance · design_effect", "Estimates and their variance"),
        ("spatial_balance · coverage_metrics · compare_designs", "Evaluate designs"),
    ]),
    ("Space-time", "several", [
        ("esda.Moran_Diff", "Autocorrelation of change between periods"),
        ("cluster.getis_ord_gi · esda.fdr_bh", "Emerging hot spots per period"),
        ("pointpattern.kde", "Space-time point densities"),
        ("interpolate.OrdinaryKriging", "Space-time interpolation"),
        ("gwmodel.GTWR", "Geographically and temporally weighted regression"),
    ]),
]

TOPICS = [
    ("04_00", "Foundation", "3", "Concepts, theory and learning paths"),
    ("04_01", "Point Pattern Analysis", "5", "Intensity, K functions, tests"),
    ("04_02", "Spatial Autocorrelation", "6", "Weights, Moran, Geary, LISA"),
    ("04_03", "Spatial Regression", "7", "Diagnostics, model selection, SAR, SEM"),
    ("04_04", "Spatial Clustering", "8", "Hot spots, DBSCAN, scan, regions"),
    ("04_05", "Disease Mapping", "11", "SMR, empirical Bayes, BYM, MCMC"),
    ("04_06", "Geographically Weighted Models", "14", "GWR, MGWR, GW machine learning"),
    ("04_07", "Spatial Interpolation", "7", "Deterministic, covariates, areal"),
    ("04_08", "Spatial Sampling Design", "5", "Designs, variance, networks"),
    ("04_09", "Spatio-temporal Analysis", "6", "Space-time weights, hot spots, GTWR"),
    ("04_10", "Capstone Projects", "5", "Health, voting, overdose, crime, deprivation"),
]

IMPORTS = ("import geolattice.spatial_statistics as sstat<br>"
           "from geolattice.spatial_statistics.weights import Queen, KNN<br>"
           "from geolattice.spatial_statistics.esda import Moran, Moran_Local")

tut = "".join(f'<div class="t"><b>{n}</b><div>{ti}<small><em>{k} notebooks</em> · {d}</small></div></div>'
              for n, ti, k, d in TOPICS)
total = sum(int(k) for _, _, k, _ in TOPICS)

HTML = f"""<!doctype html><html><head><meta charset="utf-8"><style>{sa.CSS}</style></head><body><div id="page">
<div class="bar"></div>
<div class="upatta"><img src="data:image/png;base64,{base.UPATTA}"></div>
<div class="kick"><div class="logo"><img src="data:image/png;base64,{base.LOGO}"></div>
<div class="k">QUICK REFERENCE · DR. ZIA U. AHMED</div></div>
<h1>Spatial Statistics <span>with GeoLattice</span></h1>
<div class="sub">Key functions of <b>geolattice.spatial_statistics</b>, grouped by task, with the tutorials that teach them.</div>
<div class="imps">{IMPORTS}</div>
<div class="cols">
{sa.column("PART A · PATTERNS &amp; DEPENDENCE", "Weights, patterns, clusters &amp; rates", PATTERNS)}
{sa.column("PART B · MODELS &amp; SURFACES", "Regression, GW models, surfaces &amp; samples", MODELS)}
</div>
<div class="tutall"><h3>Tutorials · Tutorials/notebooks/04_spatial_statistics · {total} notebooks</h3><div class="tgrid">{tut}</div></div>
<div class="foot"><span>SPATIAL STATISTICS WITH GEOLATTICE · UPATTA ANALYTICS</span>
<span>tags show the subpackage: sstat.esda.Moran(...) or from geolattice.spatial_statistics.esda import Moran</span></div>
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
