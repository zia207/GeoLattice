"""Sea ice mapping from two RCM ScanSAR GRD scenes with geolattice.rsense.sar.radarsat.

The runnable core of ``docs/sea_ice_mapping_rcm_aoi.md`` (Lower St. Lawrence Estuary case
study): read the AOI window of each scene, calibrate to sigma0, remove the noise floor,
speckle filter, put both dates on one UTM grid, check how far HV rises above the noise floor,
normalize HH to a reference incidence angle, then map ice versus open water (a threshold per date), ice concentration (tenths), the change between dates, and a
least-cost route across an ice-cost surface. Ice-type classification needs training labels
(CIS charts) and is left to the guide.

Usage (from the repository root)::

    python examples/rcm_sea_ice.py SCENE1_DIR SCENE2_DIR LAND_POLYGONS [OUT_DIR]

SCENE*_DIR are unzipped RCM Level-1 GRD product folders (``imagery/``, ``metadata/``),
LAND_POLYGONS any vector file of land polygons (GSHHG, OSM), and OUT_DIR defaults to
``./rcm_sea_ice_outputs``. The AOI, grid and route end points below are the case study's;
edit them for another area.
"""

import sys
import time
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
from geolattice.rsense import sar
from geolattice.rsense.sar import radarsat as rcm
from pyproj import Transformer
from rasterio.features import geometry_mask
from scipy import ndimage as ndi
from shapely.geometry import box
from skimage.graph import route_through_array

if len(sys.argv) < 4:
    sys.exit(__doc__)
SCENES = [Path(sys.argv[1]), Path(sys.argv[2])]
LAND = Path(sys.argv[3])
OUT = Path(sys.argv[4] if len(sys.argv) > 4 else "rcm_sea_ice_outputs")
OUT.mkdir(parents=True, exist_ok=True)

AOI = (-68.351803, 48.915025, -67.329460, 49.246820)   # west, south, east, north
AOI_BUF = (-68.40, 48.87, -67.28, 49.29)               # about 5 km buffer
CRS, RES, REF_DEG = "EPSG:32619", 50.0, 35.0
START, END = (-68.13, 49.20), (-67.55, 48.93)          # near Baie-Comeau, toward Matane
t0 = time.time()

# --- 1. inspect --------------------------------------------------------------------------
dates = []
for path in SCENES:
    meta = rcm.read_rcm_metadata(rcm.find_rcm_files(path, "HH")["product_xml"])
    d = str(meta["acquisition_start"])[:10]
    dates.append(d if d not in dates else f"{d}_{len(dates) + 1}")   # keep labels unique
    print(path.name, meta["satellite"], meta["beam_mode"], meta["pass"],
          f"{meta['incidence_near']:.1f}-{meta['incidence_far']:.1f} deg")

# --- 2-3. AOI window, sigma0, noise removal, speckle filter (radar geometry) --------------
subsets, filtered = {}, {}
for d, path in zip(dates, SCENES):
    for pol in ("HH", "HV"):
        s = rcm.read_rcm_grd_subset(path, pol, bounds=AOI_BUF)
        subsets[d, pol] = s
        # ENL from the most homogeneous blocks (RCM SCLN differs from the Sentinel-1 default)
        enl = sar.ship.estimate_enl(s["data"], np.isfinite(s["data"]))
        filtered[d, pol] = s | {"data": sar.refined_lee(s["data"], enl=enl)}
        print(f"{d} {pol}: window {s['window']}, ENL {enl:.1f}")


# --- 4. geocode both dates (and incidence angle, HV noise floor) onto one grid ------------
def stack(src, key, name):
    da = sar.create_stack([src[d, key] for d in dates], dates, bounds=AOI_BUF, crs=CRS,
                          resolution=RES, name=name)
    return da.rio.clip_box(*AOI, crs="EPSG:4326")


aux = {}
for d, path in zip(dates, SCENES):
    s = subsets[d, "HV"]
    _, col0, h, w = s["window"]
    nesz = rcm.interpolate_rcm_noise(rcm.read_rcm_noise_levels(rcm.find_rcm_files(path, "HV")["noise"]), w, col0=col0)
    aux[d, "nesz"] = {"data": np.broadcast_to(10 ** (nesz / 10), (h, w)).astype("float32"), "gcps": s["gcps"]}
    aux[d, "inc"] = {"data": subsets[d, "HH"]["incidence"], "gcps": subsets[d, "HH"]["gcps"]}
hh, hv = stack(filtered, "HH", "sigma0_HH"), stack(filtered, "HV", "sigma0_HV")
inc, nesz = stack(aux, "inc", "incidence"), stack(aux, "nesz", "nesz_HV")
grid = hh.isel(time=0)

# --- 5. land mask: Natural Earth, refined with bright HV land near the coast ---------------
land = gpd.read_file(LAND).to_crs(4326).clip(box(*AOI_BUF)).to_crs(CRS)
land_ne = geometry_mask(land.geometry, out_shape=grid.shape, transform=grid.rio.transform(),
                        invert=True) if len(land) else np.zeros(grid.shape, bool)
hv_max_db = sar.to_db(hv.max("time")).values
bright = ndi.binary_opening(np.nan_to_num(hv_max_db, nan=-99) > -20.0, iterations=1)
near = ndi.distance_transform_edt(~land_ne) * RES <= 2000
lab, _ = ndi.label((bright & near) | land_ne)
ids = np.unique(lab[land_ne])
land_mask = ndi.binary_dilation(np.isin(lab, ids[ids > 0]), iterations=2)
ocean = ~land_mask & np.isfinite(hh.values).all(axis=0) & np.isfinite(hv.values).all(axis=0)

# --- 6. dB and the HV noise floor: HV is used only where it is >= 3 dB above NESZ --------
hh_db, hv_db, nesz_db = sar.to_db(hh), sar.to_db(hv), sar.to_db(nesz)
hv_ok = []
for t, d in enumerate(dates):
    margin = float(np.nanmedian((hv_db.isel(time=t) - nesz_db.isel(time=t)).values[ocean]))
    hv_ok.append(margin >= 3.0)
    print(f"{d}: median HV - NESZ {margin:.1f} dB -> HV {'used' if hv_ok[-1] else 'not used (noise-limited)'}")

# --- 7. incidence normalization of HH (HV is not normalized) -------------------------------
slopes = []
for t, d in enumerate(dates):
    v = hh_db.isel(time=t).values
    first_ice = (v > sar.threshold(v[ocean & np.isfinite(v)], method="otsu")) & ocean
    try:
        fit = rcm.fit_incidence_slope(hh_db.isel(time=t), inc.isel(time=t), mask=first_ice, min_count=5000)
    except ValueError:
        continue
    ok = -0.40 <= fit["slope"] <= -0.05 and fit["r2"] >= 0.2
    print(f"{d}: HH slope fit {fit['slope']:.3f} dB/deg, R2 {fit['r2']:.2f} -> {'accepted' if ok else 'rejected'}")
    if ok:
        slopes.append(fit["slope"])
k_hh = float(np.mean(slopes)) if slopes else -0.22
print(f"HH slope used: {k_hh:.3f} dB/deg")
hh_n = rcm.normalize_incidence(hh_db, inc, k_hh, ref_deg=REF_DEG)


# --- 10. ice versus open water (per-date HH threshold), 12. concentration, 13. change -----
def ice_concentration(ice_binary, factor):
    h, w = ice_binary.shape
    blocks = ice_binary[:h // factor * factor, :w // factor * factor]
    blocks = blocks.reshape(h // factor, factor, w // factor, factor)
    with warnings.catch_warnings():                    # all-land blocks are NaN by design
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(blocks, axis=(1, 3)) * 10    # tenths


ice, ct = [], []
for t, d in enumerate(dates):
    v = hh_n.isel(time=t).values
    thr = sar.threshold(v[ocean & np.isfinite(v)], method="otsu")
    m = ndi.binary_opening((v > thr) & ocean, iterations=1) & ocean
    note = ""
    if hv_ok[t]:
        hvv = hv_db.isel(time=t).values
        ice_hv = (hvv > sar.threshold(hvv[ocean & np.isfinite(hvv)], method="otsu")) & ocean
        note = f", HH/HV agreement {(m == ice_hv)[ocean].mean():.0%}"
    ice.append(m)
    ct.append(ice_concentration(np.where(ocean, m.astype("float32"), np.nan), int(1000 / RES)))
    print(f"{d}: HH threshold {thr:.1f} dB, ice fraction {m[ocean].mean():.2f}, "
          f"mean CT {np.nanmean(ct[-1]):.1f}/10{note}")
    grid.copy(data=hh_n.isel(time=t).values.astype("float32")).rio.to_raster(OUT / f"{d}_HH_norm_db.tif")
    grid.copy(data=np.where(ocean, m, 255).astype("uint8")).rio.to_raster(OUT / f"{d}_ice_water.tif")

dhh = (hh_n.isel(time=1) - hh_n.isel(time=0)).where(ocean)
grid.copy(data=dhh.values.astype("float32")).rio.to_raster(OUT / "HH_change_db.tif")
gained, lost = (~ice[0] & ice[1])[ocean].mean(), (ice[0] & ~ice[1])[ocean].mean()
print(f"ice gained {gained:.1%}, lost {lost:.1%} of ocean pixels")

# --- 14-15. ice-cost surface and least-cost route (ice/water version) ---------------------
to_utm = Transformer.from_crs(4326, CRS, always_xy=True)


def lonlat_to_rc(lon, lat):
    col, row = ~grid.rio.transform() * to_utm.transform(lon, lat)
    return int(row), int(col)


for t, d in enumerate(dates):
    cost = np.where(ice[t], 8.0, 1.0)                  # thin first-year ice vs open water
    cost[~ocean] = 1e6
    path, total = route_through_array(cost, lonlat_to_rc(*START), lonlat_to_rc(*END),
                                      fully_connected=True, geometric=True)
    rows, cols = np.array(path).T
    print(f"{d}: route {len(path) * RES / 1000:.1f} km, cost {total:.0f}, "
          f"ice on route {ice[t][rows, cols].mean():.0%}")

print(f"done in {time.time() - t0:.1f}s, outputs in {OUT}")
