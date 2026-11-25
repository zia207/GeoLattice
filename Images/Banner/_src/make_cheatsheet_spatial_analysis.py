"""Render the Spatial Analysis quick reference (two columns of functions, tutorials below) as one PNG.

Reuses the layout, CSS and logos of make_cheatsheet_vector_raster.py.

    python Images/Banner/_src/make_cheatsheet_spatial_analysis.py
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_cheatsheet_vector_raster as base  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "03_spatial_analysis" / "03_00_spatial_analysis_quick_reference.png"
COPY = HERE.parents[2] / "Tutorials" / "notebooks" / "03_spatial_analysis" / "03_spatial_analysis_quick_reference.png"

# Each group: (heading, module tag shown on the right of the heading, rows)
SURFACES = [
    ("Vector analysis", "sa", [
        ("count_points_in_polygon · distance_matrix", "Point counts and distances"),
        ("nearest_neighbour_analysis · mean_coordinates", "Clark Evans test, mean centre"),
        ("overlap_analysis · distance_to_nearest_hub_points", "Overlap shares, nearest hubs"),
        ("kmeans_clustering · dbscan_clustering · stdbscan_clustering", "Cluster points in space and time"),
    ]),
    ("Terrain", "sa", [
        ("slope · aspect · hillshade", "First derivatives and relief shading"),
        ("curvature · plan_curvature · profile_curvature", "Surface shape"),
        ("tpi · tri · roughness", "Topographic position and ruggedness"),
        ("twi · landform_classification", "Wetness index, landform classes"),
        ("viewshed · contours", "Visibility, contour lines"),
        ("multiscale_analysis", "Terrain metrics across window sizes"),
    ]),
    ("Surface hydrology", "sa", [
        ("fill_depressions · breach_depressions", "Condition the DEM"),
        ("d8_pointer · dinf_pointer · fd8_pointer", "Flow direction models"),
        ("d8_flow_accumulation · dinf_flow_accumulation", "Upslope contributing area"),
        ("extract_streams · stream_segmentation", "Stream networks and links"),
        ("snap_pour_points · watershed · subbasins · basins", "Delineate watersheds"),
        ("elevation_above_stream · wetness_index", "HAND, topographic wetness"),
    ]),
    ("Multidimensional rasters", "sa", [
        ("MultidimensionalRaster · subset_multidimensional_raster", "NetCDF and Zarr cubes by variable and time"),
        ("aggregate_multidimensional_raster · generate_multidimensional_anomaly", "Temporal aggregates, anomalies"),
        ("generate_trend_raster · predict_using_trend_raster", "Per-pixel trends and forecasts"),
        ("analyze_changes_using_ccdc · analyze_changes_using_landtrendr", "Change detection"),
        ("multidimensional_principal_components", "PCA across time slices"),
        ("temporal_profile · sample_multidimensional", "Time series at points"),
    ]),
    ("Multi-criteria suitability", "sa", [
        ("suitability_reclassify · minmax_normalize · rescale_by_function", "Put criteria on one scale"),
        ("weighted_overlay · weighted_sum · weighted_linear_combination", "Weighted overlay and WLC"),
        ("fuzzy_membership · fuzzy_overlay · weighted_fuzzy_overlay", "Fuzzy logic overlay"),
        ("linear · gaussian · large · small · near", "Fuzzy membership functions"),
        ("categorical_and · limiting_criterion · criteria_count", "Constraints, weakest criterion, k of n"),
        ("monte_carlo_weights · perturb_weights · score_contributions", "Weight sensitivity and uncertainty"),
        ("locate_regions · top_quantile_mask", "Turn a surface into candidate sites"),
    ]),
]

PEOPLE = [
    ("Network analysis", "sna", [
        ("load_road_graph · lines_to_graph · graph_from_edgelist", "Graphs from OSM, lines or edge lists"),
        ("shortest_routes · route_summary", "Routing"),
        ("od_travel_time_matrix · travel_time_matrix", "Origin destination matrices"),
        ("isochrone_polygons · cumulative_accessibility · two_sfca", "Service areas and accessibility"),
        ("nearest_facility_network · evaluate_candidate_sites", "Facility location, p-median"),
        ("population_coverage · closure_impact", "Coverage, road closure scenarios"),
        ("fit_gravity_model · flow_communities · flow_pagerank", "Flows, regions, flow centrality"),
        ("kansky_indices · percolation_curve", "Connectivity and resilience"),
        ("simulate_contagion · metapopulation_sir", "Spread on networks"),
    ]),
    ("Segregation", "seg", [
        ("compute · SegregationResult", "Any index with confidence interval"),
        ("dissimilarity · isolation · exposure · gini · theil_h", "Single group indices"),
        ("multigroup_dissimilarity · simpson_diversity", "Multigroup indices"),
        ("local_quotient · local_entropy · local_significance", "Local measures with FDR"),
        ("kernel_weights · knn_weights · network_weights", "Neighbourhood definitions"),
        ("compute_profile", "Multiscale segregation profile"),
        ("evenness_null · systematic_null · spatially_constrained_null", "Null models"),
        ("spatial_block_bootstrap · panel_trend_test", "Inference"),
        ("decompose_change · compute_panel", "Change decomposition, panels"),
    ]),
    ("Spatial causal inference", "sc", [
        ("SpatialCausalData · SpatialPanel", "Data containers"),
        ("InterferenceEstimator · DistanceDecayExposure", "Direct and spillover effects"),
        ("diagnose_spatial_confounding · SpatialPlus · moran_eigenvectors", "Spatial confounding"),
        ("SpatialDML · SpatialDRLearner", "Double machine learning"),
        ("GWDML · GWCausalForest", "Local, varying effects"),
        ("GeographicRDD · SpatialDiD · SpatialSyntheticControl", "Quasi-experiments"),
        ("StaggeredSpatialDiD · BayesianSpatialCausal", "Staggered DiD, Bayesian models"),
        ("ClusterRandomizedDesign · power_analysis", "Design spatial experiments"),
        ("SpatialGranger · GCCM · SpatialNOTEARS", "Causal discovery"),
        ("conley_variance · block_permutation_test", "Spatial inference"),
    ]),
]

TOPICS = [
    ("03_01", "Foundation", "3", "Spatial analysis, vector and raster analysis"),
    ("03_02", "Digital Terrain Modeling", "4", "DEM preprocessing, terrain, hydrology"),
    ("03_03", "Multidimensional Data", "5", "Cubes, analysis, trends, PCA"),
    ("03_04", "Multi-Criteria Suitability", "10", "Weighted, categorical and fuzzy suitability"),
    ("03_05", "Network Analysis", "6", "Routing, service areas, flows"),
    ("03_06", "Segregation Analysis", "8", "Indices, local, multiscale, inference"),
    ("03_07", "Spatial Causal Inference", "9", "Confounding, spillovers, quasi-experiments"),
    ("03_08", "Capstone Projects", "6", "Flood, watershed, landslide, habitat, farms"),
]

IMPORTS = ("import geolattice.spatial_analysis as sa<br>"
           "from geolattice.spatial_analysis import network_analysis as sna, segregation as seg<br>"
           "import geolattice.spatial_analysis.spatialcausal as sc")

CSS = base.CSS + """
.grp h3 { display: flex; justify-content: space-between; align-items: baseline; }
.grp h3 code { font-family: 'JetBrains Mono', monospace; font-size: 9px; letter-spacing: 0; text-transform: none;
  color: #13306a; background: #8fb3ff; padding: 1px 6px; border-radius: 4px; }
.fn { grid-template-columns: 56% 44%; }
.imps { margin-top: 12px; font-family: 'JetBrains Mono', 'DejaVu Sans Mono', monospace; font-size: 9.6px; font-weight: 700;
  color: #13306a; background: #f5c518; padding: 6px 10px; border-radius: 8px; line-height: 1.55; }
.tutall { margin-top: 18px; background: rgba(10,28,72,0.42); border: 1px solid rgba(143,179,255,0.28);
  border-radius: 14px; padding: 14px 16px 12px; }
.tutall h3 { font-size: 10px; font-weight: 800; letter-spacing: 1.8px; text-transform: uppercase; color: #f5c518; margin-bottom: 8px; }
.tgrid { display: grid; grid-template-columns: 1fr 1fr; column-gap: 18px; }
.t div small em { font-style: normal; color: #f5c518; }
/* larger, wider layout for readability */
html, body { width: 1400px; }
#page { padding: 40px 48px 34px 58px; }
.bar { width: 9px; }
.upatta { top: 30px; right: 48px; padding: 8px 10px; }
.upatta img { height: 112px; }
.kick { gap: 16px; }
.logo { width: 48px; height: 48px; border-radius: 10px; padding: 3px; }
.logo img { width: 42px; height: 42px; }
.k { font-size: 14px; letter-spacing: 3.2px; }
h1 { margin-top: 16px; font-size: 46px; }
.sub { margin-top: 10px; font-size: 17px; }
.imps { margin-top: 16px; font-size: 14px; padding: 10px 14px; border-radius: 10px; }
.cols { margin-top: 24px; gap: 24px; }
.col { padding: 22px 22px 18px; border-radius: 16px; }
.colhead { margin-bottom: 12px; }
.colhead h2 { font-size: 26px; }
.colhead h2 em { font-size: 13.5px; letter-spacing: 2.6px; margin-bottom: 4px; }
.grp { margin-top: 16px; }
.grp h3 { font-size: 14px; letter-spacing: 2.2px; margin-bottom: 7px; }
.grp h3 code { font-size: 12.5px; padding: 2px 8px; }
.fn { gap: 12px; padding: 5px 0; }
.fn code { font-size: 13.4px; line-height: 1.4; }
.fn span { font-size: 14px; line-height: 1.35; }
.tutall { margin-top: 24px; padding: 20px 22px 16px; border-radius: 16px; }
.tutall h3 { font-size: 14px; letter-spacing: 2.2px; margin-bottom: 12px; }
.tgrid { column-gap: 26px; }
.t { grid-template-columns: 64px 1fr; gap: 10px; padding: 5px 0; }
.t b { font-size: 13px; padding: 3px 0; }
.t div { font-size: 16px; }
.t div small { font-size: 14px; margin-top: 2px; }
.foot { margin-top: 18px; font-size: 12px; letter-spacing: 1.8px; }
"""


def column(part, title, groups):
    g = "".join(f'<div class="grp"><h3>{h}<code>{tag}</code></h3>' + "".join(
        f'<div class="fn"><code>{n}</code><span>{d}</span></div>' for n, d in rows) + "</div>"
        for h, tag, rows in groups)
    return f'<div class="col"><div class="colhead"><h2><em>{part}</em>{title}</h2></div>{g}</div>'


tut = "".join(f'<div class="t"><b>{n}</b><div>{ti}<small><em>{k} notebooks</em> · {d}</small></div></div>'
              for n, ti, k, d in TOPICS)

HTML = f"""<!doctype html><html><head><meta charset="utf-8"><style>{CSS}</style></head><body><div id="page">
<div class="bar"></div>
<div class="upatta"><img src="data:image/png;base64,{base.UPATTA}"></div>
<div class="kick"><div class="logo"><img src="data:image/png;base64,{base.LOGO}"></div>
<div class="k">QUICK REFERENCE · DR. ZIA U. AHMED</div></div>
<h1>Spatial Analysis <span>with GeoLattice</span></h1>
<div class="sub">Key functions of <b>geolattice.spatial_analysis</b>, grouped by task, with the tutorials that teach them.</div>
<div class="imps">{IMPORTS}</div>
<div class="cols">
{column("PART A · SURFACES", "Terrain, water, cubes &amp; suitability", SURFACES)}
{column("PART B · NETWORKS &amp; PEOPLE", "Networks, segregation &amp; causality", PEOPLE)}
</div>
<div class="tutall"><h3>Tutorials · Tutorials/notebooks/03_spatial_analysis · 51 notebooks</h3><div class="tgrid">{tut}</div></div>
<div class="foot"><span>SPATIAL ANALYSIS WITH GEOLATTICE · UPATTA ANALYTICS</span>
<span>tags show the alias: sa.name(...), sna.name(...), seg.name(...), sc.Name(...)</span></div>
</div></body></html>"""

if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "sheet.html"
        src.write_text(HTML)
        raw = Path(tmp) / "raw.png"
        subprocess.run(["google-chrome", "--headless=new", "--disable-gpu", "--hide-scrollbars",
                        "--force-device-scale-factor=2", "--window-size=1400,3000",
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
