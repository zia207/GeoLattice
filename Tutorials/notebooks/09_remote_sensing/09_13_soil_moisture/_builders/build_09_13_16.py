"""Build 09_13_16_nisar_l_band_soil_moisture.ipynb: ``python _builders/run.py 16``."""

from nb import Notebook

nb = Notebook("09_13_16_nisar_l_band_soil_moisture.ipynb")

nb.md("""
# NISAR L-Band Soil Moisture

NISAR, the NASA-ISRO Synthetic Aperture Radar launched in July 2025, is the first SAR mission with
soil moisture as a standard product. Its L-band radar (24 cm wavelength) sees through crop canopies
that blind C-band Sentinel-1, and it maps every land surface twice every 12 days. NASA's Level-3
product gives soil moisture on a 200 m EASE grid, retrieved with several algorithms and merged with
quality flags. This notebook reads the provisional NISAR soil moisture over central Iowa from late
2025 to November 2026, examines its coverage and quality layers, and compares it with Sentinel-1
change detection, SMAP and the two in situ stations.
""")

nb.md("""
## Overview

### The mission

| | NISAR |
|---|---|
| Launch | 30 July 2025 (GSLV-F16, Sriharikota) |
| Radars | L-band (1.257 GHz, NASA) and S-band (3.2 GHz, ISRO) |
| Swath and repeat | 242 km, 12-day repeat, ascending and descending |
| Polarizations | quad- or dual-polarization (HH, HV, VH, VV) |
| Soil moisture product | NASA L3 SME2, 200 m EASE-Grid 2.0; ISRO L4 at 100 m over India |

### Why L-band

At L-band the canopy is more transparent and the soil contributes more of the backscatter than at
C-band, and the sensing depth is larger (about 5 cm), similar to SMAP's radiometer. Radar soil moisture
at 200 m with an L-band signal fills the gap between SMAP's 9 to 36 km and Sentinel-1's 10 m.

### The SME2 product

Each granule holds `soilMoisture` and its `soilMoistureUncertainty`, a `retrievalQualityFlag` (bit 0:
not recommended; the other bits record which algorithm contributed: PMI, a physical model inversion;
TSR, a time-series ratio method; DSG, disaggregation of SMAP), a `surfaceQualityFlag`, ancillary layers
(land cover, water fraction, local incidence angle) and the backscatter at each polarization. The
candidate retrievals of each algorithm are in `algorithmCandidates`.

The products are still maturing: the beta collection started in October 2025 and the provisional
collection used here on 28 October 2025. Check the collection version before relying on the numbers.
""")

nb.md("""
## NISAR with GeoLattice

| Function | Purpose |
|---|---|
| `sm.search_product("NISAR_SM", ...)` | granules of `NISAR_L3_SME2_PROVISIONAL_V1` in CMR |
| `sm.read_nisar_sm(file, aoi=)` | soil moisture and quality layers, only the AOI window (works on remote files) |
| `sd.nisar()` | all granules over the AOI read by HTTP range requests, cached |
| `sm.change_detection`, `sm.regrid` | Sentinel-1 retrievals on the same dates and a common grid |
| `sm.validation_metrics` | station checks |

### Learning objectives

1. Find NISAR soil moisture granules and read only the study area from them.
2. Decode the quality flag and see which algorithms produced the retrievals.
3. Compare NISAR with Sentinel-1 change detection and SMAP on common dates and grids.
4. Validate NISAR at in situ stations, keeping in mind that the product is provisional.

### Data

| Dataset | Content |
|---|---|
| NISAR L3 SME2 provisional (ASF), October 2025 to November 2026 | 200 m soil moisture, uncertainty and flags, 93.9 W to 93.2 W, 41.5 N to 42.2 N |
| Sentinel-1 RTC, relative orbit 165, October 2025 to November 2026 | VV gamma0 (dB) at 50 m; change detection references from 2019 to 2021 |
| SMAP L3 Enhanced, 2025 and 2026 | 9 km AM retrievals |
| USCRN Des Moines 17 E, SCAN Ames | in situ 5 cm |

### Contents

- Setup
- Granules and coverage
- Quality flags and algorithms
- NISAR maps
- NISAR, Sentinel-1 and SMAP on common dates
- Validation at the stations
- Summary and exercises
""")

nb.setup('''
from pyproj import Transformer  # station positions
ni = sd.nisar()  # NISAR soil moisture granules over the SAR site
print(ni.sizes, "| first", str(ni.time.values[0])[:10], "last", str(ni.time.values[-1])[:10])  # dimensions
''')

nb.md("""
## Granules and coverage
""")

nb.code('''
cov = ni.sm.notnull().mean(("y", "x")).to_series()  # share of the AOI covered by each granule
info = pd.DataFrame({"AOI covered": cov, "median sm": ni.sm.median(("y", "x")).to_series(),
                     "median uncertainty": ni.uncertainty.median(("y", "x")).to_series()})  # per granule
display(info.round(3).head(12))  # first granules
print(f"{len(info)} granules; AOI coverage per granule: median {cov.median():.0%}; "
      f"days between granules: median {np.median(np.diff(ni.time.values).astype('timedelta64[h]').astype(int)) / 24:.1f}")  # revisit
''')

nb.md("""
## Quality flags and algorithms
""")

nb.code('''
q = ni.retrieval_qual_flag.where(ni.sm.notnull())  # flags of retrieved cells
qi = q.fillna(0).astype(int)  # integer flags
bits = {"not recommended": 0, "PMI used": 1, "TSR used": 2, "DSG used": 3}  # bit meanings from the product description
share = {k: float(((qi // 2 ** b) % 2 == 1).where(q.notnull()).mean()) for k, b in bits.items()}  # share of retrievals
display(pd.Series(share).rename("share of retrievals").to_frame().T.round(3))  # table
lc = ni.land_cover.where(ni.sm.notnull())  # land cover of retrieved cells
print("retrievals by land cover code:", lc.to_series().dropna().astype(int).value_counts(normalize=True).round(3).head(6).to_dict())  # land cover
''')

nb.md("""
## NISAR maps
""")

nb.code('''
best = int(cov.values.argmax())  # granule with the largest coverage
growing = ni.sm.where(ni.time.dt.month.isin([5, 6, 7, 8, 9]))  # growing season
fig, ax = plt.subplots(1, 3, figsize=(18, 5))  # three maps
ni.sm.isel(time=best).plot(ax=ax[0], cmap="YlGnBu", vmin=0.05, vmax=0.5); ax[0].set_title(f"NISAR soil moisture, {str(ni.time.values[best])[:10]}")  # one granule
growing.mean("time").plot(ax=ax[1], cmap="YlGnBu", vmin=0.05, vmax=0.5); ax[1].set_title("Mean, May to September 2026")  # season mean
ni.uncertainty.median("time").plot(ax=ax[2], cmap="magma_r", vmin=0, vmax=0.1); ax[2].set_title("Median retrieval uncertainty")  # uncertainty
for a in ax: a.axis("off")  # clean maps
plt.tight_layout(); plt.show()  # draw
''')

nb.md("""
## NISAR, Sentinel-1 and SMAP on common dates

Sentinel-1 change detection for 2025 and 2026 uses the dry and wet references of the 2019 to 2021 record of
the same relative orbit (notebook 12). All products are averaged to the 1 km EASE grid and compared on days
when NISAR and Sentinel-1 observed within a day of each other.
""")

nb.code('''
s1_old = sd.s1_cube()  # 2019 to 2021, orbit 165: references
s1_new = sd.s1_cube(period=("2025-10-01", "2026-11-25"), name="iowa", relative_orbit=165)  # 2025 to 2026, same orbit
grow = lambda c: c.sel(time=c.time.dt.month.isin(range(4, 12)))  # unfrozen months
ref = sm.change_detection_references(grow(s1_old.vv), min_obs=20)  # dry and wet references per pixel
rel = sm.change_detection(grow(s1_new.vv), references=ref.reindex_like(s1_new.vv.isel(time=0), method="nearest"))  # relative soil moisture
soil = sd.soilgrids_grid(sd.IOWA_SAR, "iowa_sar", crs="EPSG:32615").isel(depth=0)  # texture
lim = sm.saxton_rawls(soil.sand, soil.clay, organic_matter=0.1724 * soil.soc)  # pedotransfer limits
phi = sm.regrid(lim["saturation"].rio.write_crs(soil.rio.crs), rel.isel(time=0), method="bilinear")  # porosity on the S1 grid
wp = sm.regrid(lim["wilting_point"].rio.write_crs(soil.rio.crs), rel.isel(time=0), method="bilinear")  # wilting point
cd = sm.relative_to_volumetric(rel, porosity=phi, residual=wp)  # volumetric soil moisture
g1 = sm.ease2_grid("EASE2_M01km", aoi=sd.IOWA_SAR)  # 1 km grid
tmpl = xr.DataArray(np.zeros((g1.sizes["y"], g1.sizes["x"]), "float32"), coords={"y": g1.y, "x": g1.x}, dims=("y", "x")).rio.write_crs("EPSG:6933")  # template
ni_day = pd.DatetimeIndex(ni.time.values).normalize()  # NISAR days
cd_day = pd.DatetimeIndex(cd.time.values).normalize()  # S1 days
pairs = [(i, j) for i, a in enumerate(ni_day) for j, b in enumerate(cd_day) if abs((a - b).days) <= 1]  # matched acquisitions
print(f"{len(pairs)} NISAR granules within a day of a Sentinel-1 pass")  # overlap
rows = []  # cell-level comparison at 1 km
for i, j in pairs:  # each matched pair
    a = sm.regrid(ni.sm.isel(time=i), tmpl, method="average").values.ravel()  # NISAR at 1 km
    b = sm.regrid(cd.isel(time=j), tmpl, method="average").values.ravel()  # S1 change detection at 1 km
    ok = np.isfinite(a) & np.isfinite(b)  # shared cells
    if ok.sum() > 30:  # enough cells
        rows.append({"nisar_time": ni_day[i], "cells": int(ok.sum()), "NISAR mean": a[ok].mean(), "S1 CD mean": b[ok].mean(),
                     "spatial R": np.corrcoef(a[ok], b[ok])[0, 1]})  # per pair
cmp_ = pd.DataFrame(rows)  # table
display(cmp_.round(3))  # comparison
smap = pd.concat([sm.daily_cube(sm.mask_quality(sd.ease2_clip(sd.smap_l3e(), sd.IOWA_SAR).sel(time=slice("2025-10", "2025-12")), level="all")).sm.mean(("y", "x")).to_series(),
                  sm.daily_cube(sm.mask_quality(sd.ease2_clip(sd.smap_l3e(region=sd.IOWA_REGION, period=("2026-01-01", "2026-11-24"), name="iowa"), sd.IOWA_SAR), level="all")).sm.mean(("y", "x")).to_series()])  # SMAP area mean
ser = pd.DataFrame({"NISAR (AOI mean)": ni.sm.mean(("y", "x")).to_series().groupby(ni_day).mean(),
                    "S1 change detection (AOI mean)": cd.mean(("y", "x")).to_series().groupby(cd_day).mean()})  # area means
fig, ax = plt.subplots(figsize=(13, 4))  # time series
ser.plot(ax=ax, marker="o", lw=0); smap.rolling(3, min_periods=1).mean().plot(ax=ax, color="0.5", lw=1, label="SMAP L3E (3-day mean)")  # series
ax.set_ylabel("m3/m3"); ax.legend(); ax.set_title("Area means over the SAR site"); plt.tight_layout(); plt.show()  # draw
''')

nb.md("""
## Validation at the stations
""")

nb.code('''
u26 = sd.uscrn(years=range(2025, 2027)).sm_5.resample("1D").mean()  # USCRN 5 cm, 2025 to 2026
s26 = sd.scan(start="2025-01-01", end="2026-11-24").sm_5  # SCAN 5 cm
ins = {"USCRN Des Moines 17 E": u26, "SCAN Ames": s26}  # in situ
t6933 = Transformer.from_crs(4326, 6933, always_xy=True)  # to EASE
tutm = Transformer.from_crs(4326, cd.rio.crs, always_xy=True)  # to UTM
rows = []  # metrics
for st, s in sd.STATIONS.iterrows():  # each station
    x, y = t6933.transform(s.lon, s.lat)  # EASE position
    ix, iy = int(np.abs(ni.x - x).argmin()), int(np.abs(ni.y - y).argmin())  # NISAR pixel
    p_ni = ni.sm.isel(x=slice(ix - 1, ix + 2), y=slice(iy - 1, iy + 2)).mean(("y", "x")).to_series().dropna()  # 600 m window
    p_ni = p_ni.groupby(p_ni.index.normalize()).mean()  # one value per day
    xu, yu = tutm.transform(s.lon, s.lat)  # UTM position
    p_cd = cd.sel(x=slice(xu - 300, xu + 300), y=slice(yu + 300, yu - 300)).mean(("y", "x")).to_series().dropna()  # 600 m window
    p_cd = p_cd.groupby(p_cd.index.normalize()).mean()  # per day
    for name, px in (("NISAR 200 m (600 m window)", p_ni), ("S1 change detection (600 m window)", p_cd)):  # each product
        m = sm.validation_metrics(ins[st].reindex(px.index), px)  # metrics
        rows.append({"station": st, "product": name, "n": int(m.n), "bias": m.bias, "ubrmse": m.ubrmse, "R": m.pearson_r})  # row
display(pd.DataFrame(rows).set_index(["station", "product"]).round(3))  # table
''')

nb.md("""
## Summary and exercises

- NISAR's provisional L3 soil moisture is read from ASF by HTTP range requests, only for the AOI window
  of each 200 m granule.
- The quality flag records which algorithms contributed and whether a retrieval is recommended.
- Over central Iowa NISAR can be compared with Sentinel-1 change detection on the same days at 1 km and
  with the in situ stations; the product is provisional and should be rechecked with later versions.

**Exercises**

1. Repeat the station validation with only recommended retrievals (bit 0 clear).
2. Read the candidate retrievals of each algorithm (`algorithmCandidates`) from one granule with `h5py`
   and compare them.
3. Compare the NISAR backscatter layers (`radarData/frequencyA/sigma0VV`) with Sentinel-1 VV on the same
   day: how different are L- and C-band over corn in August?
""")

nb.write()
