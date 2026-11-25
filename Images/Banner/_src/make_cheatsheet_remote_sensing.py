"""Render the Remote Sensing quick reference (two columns of functions, tutorials below) as one PNG.

Reuses the layout, CSS and logos of make_cheatsheet_spatial_analysis.py. Functions were chosen from
the calls the 09_remote_sensing notebooks make most often.

    python Images/Banner/_src/make_cheatsheet_remote_sensing.py
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
OUT = HERE.parent / "07_remote_sensing" / "09_00_remote_sensing_quick_reference.png"
COPY = HERE.parents[2] / "Tutorials" / "notebooks" / "09_remote_sensing" / "09_remote_sensing_quick_reference.png"

# Tag shown on each group -> the module it refers to (used by the name check in __main__)
MODULES = {
    "rs": "geolattice.rsense",
    "rs.access": "geolattice.rsense.access",
    "rs.optical": "geolattice.rsense.optical",
    "rs.sar": "geolattice.rsense.sar",
    "rs.insar": "geolattice.rsense.insar",
    "rs.uav": "geolattice.rsense.uav",
    "rs.hyperspectral": "geolattice.rsense.hyperspectral",
    "rs.lidar": "geolattice.rsense.lidar",
    "rs.texture": "geolattice.rsense.texture",
    "rs.super_resolution": "geolattice.rsense.super_resolution",
    "ic": "geolattice.rsense.classify.image_classification",
    "ed": "geolattice.rsense.edge_detectors",
    "fm": "geolattice.rsense.foundation_models",
    "seb": "geolattice.rsense.surface_energy_balance",
}

# Each group: (heading, tag, rows). A dotted name is relative to the tag's module.
DATA = [
    ("Core, backends and maps", "rs", [
        ("set_backend · available_backends · device", "CPU or GPU, chosen automatically"),
        ("read · write · info", "Read and write rasters as xarray"),
        ("clip · mosaic · reproject", "Basic raster preparation"),
        ("viz.plot_raster · viz.plot_rgb", "Single band and colour composite maps"),
    ]),
    ("Data access", "rs.access", [
        ("stac_search · earthdata_search", "Search STAC catalogues and NASA Earthdata"),
        ("asf_download", "Download SAR scenes from ASF"),
        ("aoi_to_geojson", "Area of interest as GeoJSON"),
    ]),
    ("Optical", "rs.optical", [
        ("load_sentinel2 · load_landsat · load_hls · load_modis", "Load scenes for an area and dates"),
        ("toa_reflectance · scale · decode_bits · landsat_qa_mask", "Calibration, scaling and QA bits"),
        ("mask_clouds · composite", "Cloud masking and compositing"),
        ("index · list_indices", "Spectral indices (NDVI, NBR, ...)"),
        ("unmix · coregister", "Spectral unmixing, co-registration"),
    ]),
    ("SAR", "rs.sar", [
        ("sentinel1.calibrate · to_db", "Calibrate to sigma0, convert to dB"),
        ("lee · refined_lee · multitemporal_speckle_filter", "Speckle filters"),
        ("multilook · terrain_correct", "Multilooking and terrain correction"),
        ("load_rtc · create_stack", "RTC products and time stacks"),
        ("polsar.h_alpha_classify", "H-alpha polarimetric classes"),
        ("ship.cfar_tiled", "CFAR ship detection"),
    ]),
    ("InSAR", "rs.insar", [
        ("differential_interferogram · flat_earth_phase", "Interferograms"),
        ("goldstein_filter · unwrap", "Filter and unwrap phase"),
        ("phase_to_los · phase_to_height", "Phase to displacement or height"),
        ("sbas_inversion · velocity · ps_candidates", "SBAS time series and PS points"),
        ("atmospheric_filter", "Remove atmospheric delay"),
    ]),
    ("UAV", "rs.uav", [
        ("flight_plan · build_catalog · read_frame", "Plan flights, catalogue and read imagery"),
        ("vegetation_indices · canopy_cover · exg_plant_mask", "Crop vigour and cover"),
        ("count_plants · tiled_detect · detect_stages", "Count and detect plants"),
        ("trait_table · health_summary", "Per-plot traits and health"),
    ]),
    ("Hyperspectral", "rs.hyperspectral", [
        ("read_envi · read_tanager · wavelengths · band_at", "Read cubes, pick bands by wavelength"),
        ("drop_bad_bands · destripe · smooth", "Clean the cube"),
        ("mnf · mnf_denoise · pca", "Dimension reduction and denoising"),
        ("continuum_removal · band_depth · red_edge_position", "Absorption features"),
        ("extract_endmembers · unmix", "Endmembers and unmixing"),
        ("rx · matched_filter · ace · cem", "Anomaly and target detection"),
    ]),
    ("LiDAR", "rs.lidar", [
        ("read_las · write_las · las_info", "Point cloud I/O"),
        ("smrf · csf", "Ground classification"),
        ("dem · dsm · chm · normalize_heights", "Terrain, surface and canopy models"),
        ("grid_metrics · detect_trees · building_footprints", "Forest and urban products"),
        ("hillshade · slope", "Terrain derivatives"),
    ]),
]

LEARNING = [
    ("Texture", "rs.texture", [
        ("glcm_texture · glrlm_texture", "Grey level co-occurrence and run length"),
        ("lbp_texture · first_order_stats · edge_density", "Local patterns and statistics"),
        ("fourier_texture · gabor_texture · wavelet_texture", "Transform based texture"),
        ("fractal_dimension · lacunarity · mrf_texture_energy", "Fractal and model based texture"),
        ("deep_texture_features", "CNN texture features"),
    ]),
    ("Super-resolution", "rs.super_resolution", [
        ("upsample · unsharp_mask · iterative_back_projection", "Classical upsampling"),
        ("pansharpen", "Pan-sharpening"),
        ("build_sr_model · train_super_resolution", "Deep super-resolution models"),
        ("train_super_resolution_gan · train_super_resolution_diffusion", "GAN and diffusion training"),
        ("predict_super_resolution", "Apply a trained model"),
    ]),
    ("Image classification", "ic", [
        ("spatial_block_split · group_split · check_split_isolation", "Leakage safe spatial splits"),
        ("sample_pixels · train_pixel_classifier · unsupervised_classify", "Pixel classification"),
        ("build_cnn · build_segmenter · train_model", "CNNs and segmentation"),
        ("predict_map · write_class_map", "Map the whole scene"),
        ("classification_report", "Accuracy assessment"),
    ]),
    ("Object detection and boundaries", "ic + ed", [
        ("ic.build_object_detector · ic.detection_tiles", "Detectors and tiling"),
        ("ic.merge_detections · ic.nms", "Merge overlapping boxes"),
        ("ic.mean_average_precision · ic.box_iou", "Detection accuracy"),
        ("ed.classical_edges · ed.build_edge_model", "Field boundary edges"),
        ("ed.detect_fields_ftw · ed.boundary_to_instances", "Field delineation to polygons"),
    ]),
    ("Foundation models", "fm", [
        ("list_models · model_info · load_model", "Prithvi, Clay, DOFA and more"),
        ("input_spec · prepare · chip_raster", "Prepare inputs for a model"),
        ("embedding_rgb · embedding_change", "Map and compare embeddings"),
        ("linear_probe", "Train a light classifier on embeddings"),
    ]),
    ("Surface energy balance", "seb", [
        ("albedo · net_radiation · soil_heat_flux", "Radiation balance"),
        ("aerodynamic_resistance · friction_velocity", "Turbulent transfer"),
        ("asce_pm_hourly", "ASCE reference ET"),
        ("run_sebal · run_metric · run_ssebop · ssebi", "Single source ET models"),
        ("priestley_taylor_jpl · tseb_pt · triangle_method", "PT-JPL, two source and triangle"),
        ("thermal_sharpen · seasonal_et · compare_to_tower", "Sharpen, upscale, validate"),
    ]),
    ("Time series, fusion and explainability", "rs", [
        ("timeseries.whittaker · timeseries.phenology", "Smoothing and phenology"),
        ("fusion.upscale", "Aggregate fine imagery to a coarse grid"),
        ("xai.aoa.dissimilarity_index", "Area of applicability"),
        ("scale.to_zarr_cube", "Large cubes as Zarr"),
    ]),
]

TOPICS = [
    ("09_01", "Foundations", "5", "Concepts, data access, backends"),
    ("09_02", "Optical", "7", "Landsat, Sentinel-2, indices, composites"),
    ("09_03", "Radar", "24", "SAR, speckle, InSAR, NISAR, ships"),
    ("09_04", "UAV", "10", "Flight planning, plant counts, traits"),
    ("09_05", "Hyperspectral", "12", "Cubes, MNF, unmixing, detection"),
    ("09_06", "LiDAR", "8", "Ground, DEM, CHM, trees, buildings"),
    ("09_07", "Texture Analysis", "6", "GLCM, LBP, Fourier, fractal, deep"),
    ("09_08", "Image Super-resolution", "5", "Classical, CNN, GAN, diffusion"),
    ("09_09", "Image Classification", "14", "Pixel, object, deep learning, accuracy"),
    ("09_10", "Object Detection", "5", "Detectors, field boundaries"),
    ("09_11", "Foundation Models", "14", "Embeddings, probes, change"),
    ("09_12", "Surface Energy Balance", "13", "SEBAL, METRIC, SSEBop, TSEB"),
    ("09_13", "Capstone Projects", "16", "End-to-end remote sensing projects"),
]

IMPORTS = ("import geolattice.rsense as rs<br>"
           "import geolattice.rsense.classify.image_classification as ic<br>"
           "import geolattice.rsense.foundation_models as fm<br>"
           "import geolattice.rsense.surface_energy_balance as seb<br>"
           "from geolattice.rsense import edge_detectors as ed")

tut = "".join(f'<div class="t"><b>{n}</b><div>{ti}<small><em>{k} notebooks</em> · {d}</small></div></div>'
              for n, ti, k, d in TOPICS)
total = sum(int(k) for _, _, k, _ in TOPICS)

HTML = f"""<!doctype html><html><head><meta charset="utf-8"><style>{sa.CSS}</style></head><body><div id="page">
<div class="bar"></div>
<div class="upatta"><img src="data:image/png;base64,{base.UPATTA}"></div>
<div class="kick"><div class="logo"><img src="data:image/png;base64,{base.LOGO}"></div>
<div class="k">QUICK REFERENCE · DR. ZIA U. AHMED</div></div>
<h1>Remote Sensing <span>with GeoLattice</span></h1>
<div class="sub">Key functions of <b>geolattice.rsense</b>, grouped by task, with the tutorials that teach them.</div>
<div class="imps">{IMPORTS}</div>
<div class="cols">
{sa.column("PART A · DATA &amp; SENSORS", "Optical, radar, UAV, hyperspectral &amp; LiDAR", DATA)}
{sa.column("PART B · ANALYSIS &amp; LEARNING", "Texture, deep learning &amp; ET models", LEARNING)}
</div>
<div class="tutall"><h3>Tutorials · Tutorials/notebooks/09_remote_sensing · {total} notebooks</h3><div class="tgrid">{tut}</div></div>
<div class="foot"><span>REMOTE SENSING WITH GEOLATTICE · UPATTA ANALYTICS</span>
<span>tags show where to call it: rs.optical.index(...), ic.train_model(...), seb.run_metric(...)</span></div>
</div></body></html>"""


def check_names():
    """Return the names on the card that do not resolve to an attribute."""
    import importlib

    bad = []
    for groups in (DATA, LEARNING):
        for _, tag, rows in groups:
            for names, _ in rows:
                for name in names.split(" · "):
                    path = MODULES[tag] if tag != "ic + ed" else None
                    if path is None:
                        prefix, name = name.split(".", 1)
                        path = MODULES[prefix]
                    *subs, attr = name.split(".")
                    try:  # dotted names are submodules of the tag's module
                        mod = importlib.import_module(".".join([path, *subs]))
                        if not hasattr(mod, attr):
                            bad.append(f"{tag}: {name}")
                    except Exception:
                        bad.append(f"{tag}: {name}")
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
                        "--force-device-scale-factor=2", "--window-size=1400,4200",
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
