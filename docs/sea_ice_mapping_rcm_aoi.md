# Sea Ice Mapping for Shipping Routes Using RADARSAT Constellation Mission (RCM) Imagery

**Case study: Lower St. Lawrence Estuary (Baie-Comeau to Matane crossing)**

**Processing library:** `geolattice.rsense.sar.radarsat` (GeoLattice `src/rsense/sar/radarsat.py`), with the sensor-generic helpers in `geolattice.rsense.sar` and `geolattice.rsense.texture`

**Worked notebook (real data):** `Tutorials/notebooks/07_remote_sensing/07_10_Capstones/07_09_09_sea_ice_mapping_RADARSAT.ipynb` runs this workflow on the two scenes with outputs.

**Runnable script:** `examples/rcm_sea_ice.py` runs Steps 1 to 7, ice–water mapping, concentration, change and an ice/water least-cost route end to end (ice-type classification needs CIS training labels and stays in this guide).

---

## 1. Overview

This workflow maps sea ice type, ice concentration, and ice change from two dual-polarization RCM ScanSAR Low Noise scenes, then converts the ice maps into an ice-cost surface and a least-cost shipping route across the AOI.

All radiometric and geometric processing runs in Python with GeoLattice. The RCM-specific steps (finding the files, calibration, thermal noise removal, ground control points, incidence angle) are in `geolattice.rsense.sar.radarsat`. Speckle filtering, geocoding to a common grid, dB conversion, texture and change detection use functions GeoLattice already shares across Sentinel-1, RCM and NISAR. ESA SNAP is optional and only useful as a visual cross-check.

### 1.1 Input scenes

| Item | Scene 1 | Scene 2 |
|---|---|---|
| Product ID | `RCM1_OK3695251_PK4015454_1_SCLNA_20260128_104701_HH_HV_GRD` | `RCM3_OK3695270_PK4023614_1_SCLND_20260203_103123_HH_HV_GRD` |
| Satellite | RCM-1 | RCM-3 |
| Date / time (UTC) | 2026-01-28, 10:47:01 | 2026-02-03, 10:31:23 |
| Beam mode | ScanSAR Low Noise, beam A (SCLNA) | ScanSAR Low Noise, beam D (SCLND) |
| Resolution / swath | 100 m / 350 km | 100 m / 350 km |
| Polarization | HH + HV | HH + HV |
| Product level | GRD (ground range detected) | GRD |

Notes:

- The two scenes use **different beams (A and D)**, so they cover different incidence angle ranges over the AOI. Incidence angle normalization (Step 7) is required before the two dates can be compared. Step 1 reads the beam mode and incidence range from each `product.xml` so you can confirm this.
- The scenes are **6 days apart**. Estuary ice drifts too far in 6 days for reliable feature tracking, so this workflow uses **ice change detection** rather than ice motion retrieval.
- Both dates fall in mid-winter, when ice is cold and dry and SAR contrast between ice types is high.

### 1.2 AOI

From `aoi.kml` ("Segregation area"):

| Bound | Value |
|---|---|
| West | -68.351803 |
| East | -67.329460 |
| South | 48.915025 |
| North | 49.246820 |

The AOI covers the middle of the Lower St. Lawrence Estuary, between Baie-Comeau on the north shore and the Matane area on the south shore, about 75 km east-west and 37 km north-south. It includes the main shipping channel and the Baie-Comeau to Matane ferry crossing.

### 1.3 Expected ice conditions

- Seasonal ice only. No multi-year ice is expected.
- Ice classes: open water, new ice and nilas, grey and grey-white ice, thin first-year ice, brash and rafted/ridged ice, and fast ice in sheltered bays.
- Ice is highly mobile. Tides, currents, and northwest winds push ice down the estuary and against the south shore.
- Wind-roughened open water is the main source of classification error.

---

## 2. Data Requirements

| Data | Source | Purpose |
|---|---|---|
| RCM scenes (2) | EODMS | Primary input |
| AOI | `aoi.kml` | Subset and analysis extent |
| Coastline / land mask | GSHHG full resolution, or OSM land polygons | Mask land |
| CIS ice charts (Gulf of St. Lawrence, 2026-01-28 and 2026-02-03) | Canadian Ice Service archive (SIGRID-3 shapefiles) | Training labels and validation |
| Wind (10 m) | ERA5 hourly, or Environment Canada | Flag wind-roughened water |
| AIS ship tracks | Commercial AIS provider or MarineCadastre-style archive | Route validation |
| Optical imagery (optional) | Sentinel-2, MODIS | Visual validation on clear days |

A DEM is not needed. Over open water and sea ice the surface is at sea level, so geocoding with the product's ellipsoid tie points (Step 6) is accurate. Terrain displacement only affects the steep shoreline, which the coastal buffer in the land mask removes anyway.

---

## 3. Software Environment

```bash
conda create -n seaice python=3.11 -y
conda activate seaice
conda install -c conda-forge gdal rasterio rioxarray xarray geopandas shapely \
    scikit-learn scikit-image scipy numpy pandas matplotlib fiona pyproj -y

# GeoLattice (provides geolattice.rsense.sar.radarsat), installed from your local clone
pip install -e /path/to/GeoLattice

# Optional: ESA SNAP 9 or later, only for visual cross-checks
```

```python
import numpy as np
import xarray as xr
from geolattice.rsense import sar
from geolattice.rsense.sar import radarsat as rcm
from geolattice.rsense import texture
```

### 3.1 Project structure

```
seaice_rcm/
├── data/
│   ├── raw/
│   │   ├── RCM1_OK3695251_PK4015454_1_SCLNA_20260128_104701_HH_HV_GRD/
│   │   └── RCM3_OK3695270_PK4023614_1_SCLND_20260203_103123_HH_HV_GRD/
│   ├── aux/            # aoi.kml, coastline, CIS charts, ERA5, AIS
│   └── processed/      # calibrated, filtered, geocoded rasters
├── outputs/
│   ├── maps/
│   ├── figures/
│   └── tables/
└── notebooks/
```

### 3.2 RCM GRD product layout

Each unzipped RCM product folder contains:

```
<PRODUCT>/
├── imagery/
│   ├── <PRODUCT>_HH.tif        # one GeoTIFF per polarization, amplitude DN (uint16)
│   └── <PRODUCT>_HV.tif
├── metadata/
│   ├── product.xml             # scene metadata and the tie-point (geolocation) grid
│   └── calibration/
│       ├── lutSigma_HH.xml     # sigma-nought gain LUT, one per polarization
│       ├── lutSigma_HV.xml
│       ├── lutBeta_<pol>.xml
│       ├── lutGamma_<pol>.xml
│       ├── noiseLevels_HH.xml  # NESZ by range, one segment per ScanSAR sub-beam
│       ├── noiseLevels_HV.xml
│       └── incidenceAngles.xml # ellipsoid incidence angle by range
└── preview/
```

`rcm.find_rcm_files(product_dir, pol)` locates these files for one polarization and returns `None` for any that are missing. It also accepts older products that use unsuffixed `lutSigma.xml` / `noiseLevels.xml`.

### 3.3 Functions used

| Step | Function | Module |
|---|---|---|
| Inspect | `find_rcm_files`, `read_rcm_metadata`, `read_rcm_tie_points` | `sar.radarsat` |
| Subset + calibrate + noise | `read_rcm_grd_subset` (uses `read_rcm_lut`, `interpolate_rcm_lut`, `read_rcm_noise_levels`, `interpolate_rcm_noise`, `calibrate_rcm`) | `sar.radarsat` |
| Incidence angle | `read_rcm_incidence`, `rcm_incidence_angle` | `sar.radarsat` |
| Speckle filter | `refined_lee`, `lee`, `enl_estimate` | `sar.speckle` |
| Geocode to common grid | `create_stack`, `common_grid` | `sar.sentinel1` (sensor-generic) |
| dB conversion | `to_db`, `from_db` | `sar.sentinel1` |
| Incidence normalization | `fit_incidence_slope`, `normalize_incidence` | `sar.radarsat` |
| Texture | `glcm_texture`, `first_order_stats` | `rsense.texture` |
| Ice–water threshold | `threshold` | `sar.change` |
| Change | `log_ratio` | `sar.change` |

All of these are also available flat on `geolattice.rsense.sar` (for example `sar.read_rcm_grd_subset`).

---

## 4. Step-by-Step Workflow

```python
SCENES = {
    "2026-01-28": "data/raw/RCM1_OK3695251_PK4015454_1_SCLNA_20260128_104701_HH_HV_GRD",
    "2026-02-03": "data/raw/RCM3_OK3695270_PK4023614_1_SCLND_20260203_103123_HH_HV_GRD",
}
AOI = (-68.351803, 48.915025, -67.329460, 49.246820)   # west, south, east, north
AOI_BUF = (-68.40, 48.87, -67.28, 49.29)               # about 5 km buffer
CRS, RES = "EPSG:32619", 50.0                          # UTM 19N, 50 m
```

### Step 1. Inspect the scenes

```python
for date, path in SCENES.items():
    files = rcm.find_rcm_files(path, "HV")
    meta = rcm.read_rcm_metadata(files["product_xml"])
    print(date, meta["satellite"], meta["beam_mode"], meta["pass"], meta["polarizations"],
          f"{meta['incidence_near']:.1f}-{meta['incidence_far']:.1f} deg",
          f"{meta['pixel_spacing']} m", meta["pixel_time_ordering"])
    print("  noise file:", files["noise"], " incidence file:", files["incidence"])
```

Confirm:

- Both HH and HV images and their `lutSigma_<pol>.xml` and `noiseLevels_<pol>.xml` files are present.
- The AOI falls inside the swath. `read_rcm_grd_subset` (Step 2) raises `ValueError` if it does not.
- The incidence angle range over the AOI. Record it for each date (Step 2 returns it per pixel).

Also check the preview image for obvious artifacts (scalloping, beam seams, radio interference), and record ERA5 wind speed and direction at about 10:30 to 11:00 UTC for each date.

### Step 2. Subset, calibrate to sigma-nought and remove thermal noise

One call per scene and polarization does three things on the AOI window only:

1. Reads the window covering `AOI_BUF` (plus a margin of 100 pixels) from the GeoTIFF.
2. Calibrates to σ⁰ with the product's own gain LUT.
3. Subtracts the noise-equivalent sigma-nought (NESZ) for that polarization.

```python
subsets = {}
for date, path in SCENES.items():
    for pol in ("HH", "HV"):
        subsets[date, pol] = rcm.read_rcm_grd_subset(
            path, pol, bounds=AOI_BUF, kind="sigma", noise=True, margin=100)

s = subsets["2026-01-28", "HV"]
print(s["window"], s["data"].shape, np.nanmedian(s["data"]), s["incidence"].min(), s["incidence"].max())
```

Each result is a dict:

| Key | Content |
|---|---|
| `data` | σ⁰, linear power, float32, still in radar geometry (rows = azimuth, columns = range) |
| `gcps`, `gcp_crs` | ground control points around the window, shifted to window coordinates |
| `window` | `(row0, col0, height, width)` of the window in the full scene |
| `incidence` | ellipsoid incidence angle in degrees, same shape as `data` |
| `meta` | the `read_rcm_metadata` dict |

Calibration. For a pixel at range column *j*:

```
σ⁰(j) = (DN(j)² + B) / A(j)
```

`A(j)` is the gain from `lutSigma_<pol>.xml`, interpolated along range (the LUT is constant along azimuth), and `B` is the LUT offset. RCM gains are used as stored, not squared. `calibrate_rcm` sets DN = 0 (no data) to NaN.

Thermal noise. The HV channel over open water and thin ice is close to the instrument noise floor. Low Noise mode reduces this problem but does not remove it. NESZ is stored in `noiseLevels_<pol>.xml` in dB, as one segment per ScanSAR sub-beam. `interpolate_rcm_noise` stitches the segments across range (filling sub-beam seams from the neighbouring segment ends), and the noise is subtracted in linear σ⁰. Values that fall below zero are clipped to a small floor (1e-6). Use `noise=False` to keep the raw σ⁰, for example to compare HH with and without noise removal.

Ground control points come from the GeoTIFF when it carries them. Otherwise they are read from the tie-point grid in `product.xml` (`read_rcm_tie_points`).

> **Lower-level route.** The same steps can be called one by one, for example to inspect the noise profile:
>
> ```python
> files = rcm.find_rcm_files(SCENES["2026-01-28"], "HV")
> lut = rcm.read_rcm_lut(files["lut_sigma"])
> segs = rcm.read_rcm_noise_levels(files["noise"], cal_type="Sigma Nought")
> row0, col0, h, w = s["window"]
> gain = rcm.interpolate_rcm_lut(lut, w, col0=col0)
> nesz_db = rcm.interpolate_rcm_noise(segs, w, col0=col0)   # plot this against column
> ```

### Step 3. Speckle filtering (radar geometry)

Filter before geocoding, while neighbouring pixels are still independent looks.

```python
# ENL for RCM SCLN differs from the Sentinel-1 default (4.4) used by the filters.
# Estimate it from the most homogeneous 64x64 blocks of the window:
d = subsets["2026-01-28", "HH"]["data"]
enl = sar.ship.estimate_enl(d, np.isfinite(d))
# or over a box you know is homogeneous (calm open water or uniform level ice):
# enl = sar.enl_estimate(d[200:300, 400:500])

filtered = {k: v | {"data": sar.refined_lee(v["data"], enl=enl)} for k, v in subsets.items()}
```

- `sar.refined_lee` (7×7 edge-aligned windows) preserves floe edges and leads. `sar.lee(img, size=5, enl=enl)` is a faster alternative.
- Keep the unfiltered `subsets` for texture (Step 8), because filtering removes useful texture.

### Step 4. Geocode both dates onto one map grid

`create_stack` geocodes each scene with its own ground control points directly onto a common UTM 19N grid. Because every date lands on identical pixels, the two dates are co-registered by construction, even though they come from different beams and orbits.

```python
dates = list(SCENES)

def stack(src, pol):
    return sar.create_stack([src[d, pol] for d in dates], dates,
                            bounds=AOI_BUF, crs=CRS, resolution=RES, name=f"sigma0_{pol}")

hh, hv = stack(filtered, "HH"), stack(filtered, "HV")          # (time, y, x), linear
hh_raw, hv_raw = stack(subsets, "HH"), stack(subsets, "HV")    # unfiltered, for texture

inc = sar.create_stack([{"data": subsets[d, "HH"]["incidence"], "gcps": subsets[d, "HH"]["gcps"]}
                        for d in dates], dates, bounds=AOI_BUF, crs=CRS, resolution=RES,
                       name="incidence")

clip = lambda da: da.rio.clip_box(*AOI, crs="EPSG:4326")
hh, hv, hh_raw, hv_raw, inc = map(clip, (hh, hv, hh_raw, hv_raw, inc))
hh.isel(time=0).rio.to_raster("data/processed/20260128_Sigma0_HH.tif")
```

- Pixel spacing: the SCLN GRD products are sampled at **40 m** (100 m resolution); a **50 m** grid keeps their detail while halving the pixel count. Use 100 m for a lighter run.
- Geocoding uses the tie points on the ellipsoid, not a DEM (see Section 2).
- If you prefer SNAP's Range-Doppler Terrain Correction, export SNAP's terrain-corrected σ⁰ and continue at Step 5. The results over water should agree within a pixel.

### Step 5. Land mask

```python
import geopandas as gpd
from rasterio.features import geometry_mask
from scipy import ndimage as ndi
from shapely.geometry import box

land = gpd.read_file("data/aux/land_polygons.shp").to_crs(4326).clip(box(*AOI_BUF)).to_crs(CRS)
grid = hh.isel(time=0)
land_mask = geometry_mask(land.geometry, out_shape=grid.shape,
                          transform=grid.rio.transform(), invert=True)
land_mask = ndi.binary_dilation(land_mask, iterations=2)    # 1 to 2 pixel coastal buffer
ocean = ~land_mask & np.isfinite(grid.values)
```

Masking can also be done before geocoding, in radar geometry, with `sar.ship.land_mask_for_window(s["gcps"], s["data"].shape, land_geoms_lonlat, buffer_m=100, pixel_spacing=50)`. That is useful if you run CFAR ship detection on the same windows.

### Step 6. Convert to dB

```python
hh_db, hv_db = sar.to_db(hh), sar.to_db(hv)
```

`to_db` applies a small floor before the logarithm, so noise-clipped pixels do not become `-inf`.

### Step 7. Incidence angle normalization

HH backscatter decreases with incidence angle (θ). Because the two dates use different beams, normalize both to a reference angle (θref = 35°):

```
σ⁰_norm(dB) = σ⁰(dB) - k × (θ - θref)
```

Estimate the slope `k` from your own data over a large area of uniform ice, then apply it:

```python
# uniform_ice: boolean mask of a large homogeneous first-year ice area (from Step 9 labels
# or drawn by hand), on the same grid as hh_db
fit = rcm.fit_incidence_slope(hh_db.isel(time=0), inc.isel(time=0), mask=uniform_ice & ocean)
print(fit)    # {'slope': ..., 'intercept': ..., 'r2': ..., 'n': ...}

k = fit["slope"] if (-0.40 <= fit["slope"] <= -0.05 and fit["r2"] >= 0.2) else -0.22
hh_n = rcm.normalize_incidence(hh_db, inc, k, ref_deg=35.0)
hv_n = hv_db                       # HV is not normalized (see below)
```

- Typical slope for first-year ice in HH: k ≈ -0.20 to -0.25 dB/degree. A fitted slope far outside this range usually means the mask mixes surface types.
- Fit each date separately and compare. If the slopes agree, use their mean for both dates. Within one scene the AOI spans only a few degrees and ice types vary across it, so the fit is often weakly constrained; the code falls back to -0.22 unless the fit is plausible (-0.40 to -0.05) with R² ≥ 0.2. On the two case-study scenes both fits were rejected (R² 0.07 and 0.14).
- Do not normalize HV. Its incidence dependence over ice is small, and on a noise-limited scene a fitted HV slope describes the noise floor, not the ice (on the 3 February scene it came out at about +2 dB/degree).
- **Check the HV noise floor first.** Geocode the NESZ (`rcm.interpolate_rcm_noise`, then `create_stack`) and compare it with noise-corrected HV over the ocean. Use HV only on scenes where the median margin is at least 3 dB. In the case study, HV was 3.8 dB above the floor on 28 January (32–38° incidence) and 0.1 dB above it on 3 February (50–54°).
- `normalize_incidence` keeps the xarray coordinates and records `incidence_reference_deg` and `incidence_slope_db_per_deg` in `attrs`.

### Step 8. Feature extraction

Build a feature stack for each date.

| Feature | Description | Source |
|---|---|---|
| `HH_dB` | Normalized HH backscatter | Step 7 |
| `HV_dB` | Noise-corrected HV backscatter | Step 7 |
| `ratio_dB` | HH_dB - HV_dB (co-/cross-pol ratio) | arithmetic |
| `GLCM_contrast_HH` | Texture: roughness, ridges, floe edges | `texture.glcm_texture` |
| `GLCM_homogeneity_HH` | Texture: smooth new ice versus rough ice | `texture.glcm_texture` |
| `GLCM_entropy_HV` | Texture: structure in cross-pol | `texture.glcm_texture` |
| `local_std_HH` | Local standard deviation (7×7 window) | `texture.first_order_stats` |
| `dist_coast` | Distance to shore (helps identify fast ice) | `scipy.ndimage` |

```python
def features(t):
    hh_t, hv_t = hh_n.isel(time=t), hv_n.isel(time=t)
    hh_tex = sar.to_db(hh_raw.isel(time=t))           # texture on the unfiltered image
    hv_tex = sar.to_db(hv_raw.isel(time=t))
    g_hh = texture.glcm_texture(hh_tex, size=11, step=2, levels=32,
                                props=("contrast", "homogeneity"))
    g_hv = texture.glcm_texture(hv_tex, size=11, step=2, levels=32, props=("entropy",))
    std_hh = texture.first_order_stats(hh_tex, size=7, stats=("std",))["std"]
    dist = ndi.distance_transform_edt(~land_mask) * RES / 1000.0     # km
    layers = [hh_t, hv_t, hh_t - hv_t, g_hh["contrast"], g_hh["homogeneity"],
              g_hv["entropy"], std_hh, dist]
    return np.stack([np.asarray(getattr(a, "values", a), "float32") for a in layers])

FEATURES = ["HH_dB", "HV_dB", "ratio_dB", "GLCM_contrast_HH", "GLCM_homogeneity_HH",
            "GLCM_entropy_HV", "local_std_HH", "dist_coast"]
stack1, stack2 = features(0), features(1)          # (n_features, y, x)
```

- `glcm_texture` computes the GLCM every `step` pixels and resamples back to full resolution. `step=2` is a good speed/detail balance at 50 m. It averages the four standard directions.
- Quantization uses 32 grey levels. With an 11×11 window this populates the co-occurrence matrix reliably.
- GLCM is the slowest step. On this AOI it takes under a minute per date at 100 m and a few minutes at 50 m. Raise `step` to 3 or 4 for quick trials.

### Step 9. Training labels from CIS ice charts

1. Download CIS Gulf of St. Lawrence charts (SIGRID-3) for 2026-01-28 and 2026-02-03, or the nearest dates.
2. Read the egg code attributes: total concentration (`CT`), stage of development (`SA`, `SB`, `SC`), and partial concentrations (`CA`, `CB`, `CC`).
3. Map stage codes to analysis classes:

| Class | ID | CIS stage of development |
|---|---|---|
| Open water | 0 | Ice-free, or CT < 1/10 |
| New ice / nilas | 1 | New ice, nilas |
| Grey / grey-white ice | 2 | Grey, grey-white, young ice |
| Thin first-year ice | 3 | Thin first-year (30 to 70 cm) |
| Rough / deformed ice | 4 | Ridged, rafted, brash (from chart remarks and SAR texture) |
| Fast ice | 5 | Fast ice polygons |

4. Use only polygons where one stage dominates (partial concentration of 8/10 or more) to select training pixels.
5. Refine labels by visual interpretation of the SAR image. Chart polygons are coarse and ice moves between chart time and image time.
6. Sample pixels in a balanced way across classes (for example 2,000 to 5,000 per class) and keep separate polygons for validation.

Rasterize the selected polygons onto the Step 4 grid (`rasterio.features.rasterize` with `grid.rio.transform()`), so `labels` has the same shape as one band of `stack1`.

### Step 10. Ice–water discrimination

Separate ice from open water before classifying ice type.

```python
t = 0                                                  # repeat for each date
hh1 = hh_n.isel(time=t).values
t_hh = sar.threshold(hh1[ocean], method="otsu")        # per-date threshold on normalized HH
ice = ndi.binary_opening((hh1 > t_hh) & ocean, iterations=1) & ocean

if hv_usable[t]:                                       # HV at least 3 dB above NESZ (Step 7)
    hv1 = hv_n.isel(time=t).values
    ice_hv = (hv1 > sar.threshold(hv1[ocean], method="otsu")) & ocean
    print("HH/HV agreement:", (ice == ice_hv)[ocean].mean())
    # Wind-roughened water: called ice by HH, dark in HV, smooth texture
    contrast = stack1[FEATURES.index("GLCM_contrast_HH")]
    rough_water = ice & ~ice_hv & (contrast < np.nanmedian(contrast[ice]))
    if era5_wind_ms > 8:                               # from Step 1
        ice &= ~rough_water
```

1. Threshold each date separately. The two beams differ in incidence and noise floor, so one threshold for both dates fails (on the case-study pair, a pooled HV threshold gave 99.9% ice on one date and 1.1% on the other).
2. Use normalized HH as the primary ice indicator: it separates ice from calm water at both moderate and steep incidence. Check the Otsu value against a histogram; if it is not bimodal, set the threshold by hand from known water and ice areas.
3. Where HV is above the noise floor, use it to confirm the HH map and to catch wind-roughened water, which is bright in HH but stays dark in HV.
4. Use ERA5 wind: where wind exceeds about 8 m/s, treat the flagged pixels as open water.
5. Apply a small morphological opening to remove isolated noise pixels.

### Step 11. Ice type classification

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, cohen_kappa_score

train_rows, train_cols = np.nonzero((labels != 255) & ocean)
X = stack1[:, train_rows, train_cols].T          # (n_samples, n_features)
y = labels[train_rows, train_cols]
ok = np.isfinite(X).all(axis=1)
X, y = X[ok], y[ok]

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

rf = RandomForestClassifier(n_estimators=500, max_features="sqrt", min_samples_leaf=5,
                            class_weight="balanced", n_jobs=-1, random_state=42)
rf.fit(X_tr, y_tr)

pred = rf.predict(X_te)
print(classification_report(y_te, pred))
print("Kappa:", cohen_kappa_score(y_te, pred))
print(dict(zip(FEATURES, rf.feature_importances_.round(3))))

# Predict the full scene (valid ocean pixels only)
valid = ocean & np.isfinite(stack1).all(axis=0)
ice_class = np.full(ocean.shape, 255, np.uint8)
ice_class[valid] = rf.predict(stack1[:, valid].T)
ice_class[valid & ~ice] = 0                       # Step 10 water mask overrides
```

Recommendations:

- Train on Scene 1 and test on Scene 2 (`stack2`) to check whether the model transfers across dates and beams. Incidence normalization (Step 7) is what makes this transfer possible.
- Report feature importance to show which channels drive the classification.
- Apply a 3×3 majority filter to the classified map (for example `scipy.ndimage.generic_filter` with a mode function, or `skimage.filters.rank.majority`).
- Optional: replace Random Forest with a U-Net trained on the AI4Arctic/AutoICE dataset and fine-tuned with local labels.

### Step 12. Sea ice concentration

Aggregate the ice–water map to a coarser grid:

```python
def ice_concentration(ice_binary, factor=20):
    """ice_binary: 1 = ice, 0 = water, NaN = land. factor 20 at 50 m = 1 km cells."""
    h, w = ice_binary.shape
    h2, w2 = h // factor, w // factor
    blocks = ice_binary[:h2*factor, :w2*factor].reshape(h2, factor, w2, factor)
    return np.nanmean(blocks, axis=(1, 3)) * 10   # tenths (0 to 10)

ice_f = np.where(ocean, ice.astype("float32"), np.nan)
ct_1km, ct_5km = ice_concentration(ice_f, 20), ice_concentration(ice_f, 100)
```

- Output in **tenths** (0 to 10) to match CIS convention.
- Use a 1 km grid for route analysis and a 5 km grid for comparison with CIS charts.
- The block grid starts at the top-left corner of the Step 4 grid. Its transform is `grid.rio.transform() * Affine.scale(factor)`.

### Step 13. Change detection between dates

Both dates are already on the same grid (Step 4) and normalized to the same incidence angle (Step 7).

1. Compute:
   - **Class transition matrix** (from-to table of ice classes, Jan 28 to Feb 3)
   - **Concentration change** (ΔCT = CT_Feb3 - CT_Jan28)
   - **HH log-ratio** on the normalized images. In dB this is a difference:

```python
dhh = hh_n.isel(time=1) - hh_n.isel(time=0)                      # dB, = 10 log10(Feb3 / Jan28)
# equivalent with the library function on linear values:
dhh_lr = sar.log_ratio(sar.from_db(hh_n.isel(time=0)), sar.from_db(hh_n.isel(time=1)))

import pandas as pd
both = (ice_class_1 != 255) & (ice_class_2 != 255)
transition = pd.crosstab(ice_class_1[both], ice_class_2[both], rownames=["Jan 28"], colnames=["Feb 3"])
d_ct = ct_1km_2 - ct_1km_1
```

2. Interpret the results:
   - New ice formation in areas that were open water
   - Ice export or compaction against the south shore
   - Growth or breakup of fast ice
3. Because the scenes are 6 days apart, describe these as **net changes** in ice state, not ice drift.
4. Do not use `sar.omnibus_test` or `sar.pairwise_change` here. They assume both dates share the same acquisition geometry and number of looks, which two different beams do not.

### Step 14. Ice-cost surface for navigation

Assign a navigation cost to each pixel based on ice class and concentration.

| Class | Base cost |
|---|---|
| Open water | 1 |
| New ice / nilas | 2 |
| Grey / grey-white ice | 4 |
| Thin first-year ice | 8 |
| Rough / deformed ice | 15 |
| Fast ice | 50 (avoid) |
| Land | impassable |

```
cost = base_cost(class) × (1 + CT / 10)
```

```python
from scipy.ndimage import zoom

base = np.array([1, 2, 4, 8, 15, 50], "float32")
ct_full = zoom(np.nan_to_num(ct_1km), 20, order=0)[:ice_class.shape[0], :ice_class.shape[1]]
ct_full = np.pad(ct_full, ((0, ice_class.shape[0] - ct_full.shape[0]),
                           (0, ice_class.shape[1] - ct_full.shape[1])), mode="edge")
cost = np.full(ice_class.shape, 1e6, "float32")          # land and no data: effectively impassable
ok = ice_class != 255
cost[ok] = base[ice_class[ok]] * (1 + ct_full[ok] / 10)
```

Notes:

- Adjust costs for the ship ice class (for example a Polar Class or Baltic ice class vessel versus an icebreaker).
- POLARIS is designed for polar waters. For the St. Lawrence, a custom cost scheme like this one is easier to justify and explain.
- Add a penalty for pressure zones along the south shore if wind is from the northwest.

### Step 15. Least-cost route

```python
from skimage.graph import route_through_array
from pyproj import Transformer

to_utm = Transformer.from_crs(4326, 32619, always_xy=True)
transform = grid.rio.transform()

def lonlat_to_rc(lon, lat, transform):
    x, y = to_utm.transform(lon, lat)
    col, row = ~transform * (x, y)
    return int(row), int(col)

start = lonlat_to_rc(-68.13, 49.20, transform)   # near Baie-Comeau (adjust to open water)
end   = lonlat_to_rc(-67.55, 48.93, transform)   # toward Matane (adjust to open water)

path, total_cost = route_through_array(cost, start, end, fully_connected=True, geometric=True)
```

- Run the route for both dates and compare path length, total cost, and the ice classes crossed.
- Compare with the straight-line route to measure the detour caused by ice.
- Choose start and end points in open water near each port. Confirm them on the map before running (`cost[start]` and `cost[end]` must not be 1e6).

### Step 16. Accuracy assessment

1. **Classification accuracy**: confusion matrix, overall accuracy, per-class F1 score, and Cohen's kappa on held-out validation polygons.
2. **Concentration**: compare SAR concentration with CIS total concentration (`CT`) on the 5 km grid. Report bias, RMSE, and R².
3. **Cross-date transfer**: report accuracy when the model trained on Jan 28 is applied to Feb 3.
4. **Route validation**: overlay AIS tracks for each date. Measure how far actual ship tracks are from the modelled route, and check which ice classes ships actually crossed.
5. **Error notes**: record areas affected by wind-roughened water, ScanSAR beam seams (visible as range-parallel stripes in HV where the noise segments meet), and the coastline.

### Step 17. Outputs

| Output | Format |
|---|---|
| Calibrated, noise-corrected σ⁰ HH/HV (dB), both dates | GeoTIFF |
| Incidence angle grid, both dates | GeoTIFF |
| Ice–water maps | GeoTIFF |
| Ice type maps | GeoTIFF + PNG map |
| Ice concentration (1 km, 5 km) | GeoTIFF |
| Change maps and transition matrix | GeoTIFF + CSV |
| Ice-cost surfaces | GeoTIFF |
| Least-cost routes | GeoJSON / shapefile |
| Accuracy tables | CSV |
| Figures (maps, feature importance, route comparison) | PNG |

Write arrays back onto the Step 4 grid with `grid.copy(data=array).rio.to_raster(path)`.

Map layout suggestion: two-panel figure (Jan 28 | Feb 3) for each product, using the WMO sea ice colour scheme for ice types and a common colour bar for concentration.

---

## 5. Processing Summary

```
RCM GRD folder (HH, HV)  ×2 dates
   │
   ├─ rcm.read_rcm_grd_subset      subset to AOI (+5 km) → σ⁰ (lutSigma) → minus NESZ (noiseLevels)
   │                               + GCPs + incidence angle (incidenceAngles.xml)
   ├─ sar.refined_lee              speckle filter in radar geometry (ENL from sar.enl_estimate)
   ├─ sar.create_stack             geocode both dates to one UTM 19N, 50 m grid (ellipsoid GCPs)
   ├─ land mask, sar.to_db
   ├─ rcm.fit_incidence_slope / rcm.normalize_incidence   (θref = 35°)
   │
   ├─ Feature stack (HH, HV, ratio, texture.glcm_texture, first_order_stats, distance to coast)
   ├─ Labels from CIS charts + visual interpretation
   │
   ├─ Ice–water discrimination (sar.threshold + wind/texture rules)
   ├─ Ice type classification (Random Forest / U-Net)
   ├─ Ice concentration (1 km, 5 km)
   ├─ Change detection (Jan 28 → Feb 3)
   │
   ├─ Ice-cost surface
   ├─ Least-cost route (Baie-Comeau → Matane)
   │
   └─ Validation (CIS charts, AIS) → maps, tables, report
```

---

## 6. Limitations

- **Resolution**: 100 m resolution cannot resolve narrow leads, small floes, or individual ridges. Ridges appear only as texture.
- **Wind effects**: wind-roughened open water can be misclassified as ice, especially in HH.
- **Young ice classes**: new, grey, and grey-white ice have similar backscatter, so confusion between them is expected.
- **Different beams**: SCLNA and SCLND have different incidence angles and noise profiles. Normalization reduces but does not remove this difference.
- **Noise floor**: after NESZ subtraction, HV over calm water is close to zero and many pixels sit at the clip floor. Treat HV values below about -28 dB as "at the noise floor", not as a measured signal.
- **Geocoding**: ellipsoid tie-point geocoding is accurate over water but can be off by a pixel or two on steep coast; the coastal buffer hides this.
- **Time gap**: 6 days is too long for ice drift tracking in the estuary.
- **Label quality**: CIS chart polygons are generalized and may not match image time exactly.
- **Small AOI**: the AOI covers a small area, so the number of independent validation samples is limited.

---

## 7. Possible Extensions

- Add RCM or Sentinel-1 scenes 1 to 2 days apart to measure ice drift. Sentinel-1 GRD subsets from `sar.read_grd_subset` go into the same `sar.create_stack` call as the RCM subsets.
- Build a winter time series (December to March) to map how ice regimes evolve.
- Fuse SAR with Sentinel-2 or MODIS on clear days to improve labels.
- Train a U-Net with local labels and compare it with Random Forest.
- Add SAR ship detection and compare detections with AIS and the modelled routes: `sar.ship.land_mask_for_window`, `sar.ship.estimate_enl` and `sar.ship.cfar` run on the radar-geometry `data` from Step 2, and `sar.ship.match_points` scores detections against AIS.
- For RCM compact-pol products (or the RCM ARD RR/RL product), `rcm.rr_rl_to_stokes` and `rcm.compact_pol_mchi` add surface/volume/double-bounce features to the Step 8 stack.

---

## 8. Key References

- Canadian Ice Service. *MANICE: Manual of Standard Procedures for Observing and Reporting Ice Conditions.*
- World Meteorological Organization. *WMO Sea-Ice Nomenclature* (WMO-No. 259).
- JCOMM. *SIGRID-3: A Vector Archive Format for Sea Ice Charts.*
- Canadian Space Agency. *RADARSAT Constellation Mission Product Specification.*
- Zakhvatkina, N., et al. (2019). Classification of sea ice types in ScanSAR RADARSAT-2 images using SVM and texture features. *Remote Sensing.*
- Stokholm, A., et al. (2022). AI4SeaIce: Toward solving ambiguous SAR textures in convolutional neural networks for automatic sea ice concentration charting. *IEEE Transactions on Geoscience and Remote Sensing.*
