"""Build 09_13_04_smap_l3_enhanced_daily_cube_and_climatology.ipynb: ``python _builders/run.py 04``."""

from nb import Notebook

nb = Notebook("09_13_04_smap_l3_enhanced_daily_cube_and_climatology.ipynb")

nb.md("""
# SMAP L3 Enhanced: a Daily Cube and its Climatology

NASA's Soil Moisture Active Passive (SMAP) mission has measured L-band brightness temperature
since March 2015. Its radar failed after three months, but the radiometer has run ever since,
giving one of the most accurate global soil moisture records. The L3 Enhanced product
(SPL3SMP_E) grids each day's retrievals onto the 9 km EASE-Grid 2.0, using Backus-Gilbert
interpolation of the overlapping 36 km footprints. This notebook builds a ten-year daily cube over
Iowa and Nebraska from the AM and PM retrievals, applies the quality flags, counts what survives,
computes a smoothed day-of-year climatology, and uses it to show the 2023 drought as an anomaly.
""")

nb.md("""
## Overview

### The SMAP product family

| Product | Grid | Content |
|---|---|---|
| L2_SM_P, L3_SM_P (SPL3SMP) | 36 km | single channel retrieval from the radiometer footprint |
| L2_SM_P_E, L3_SM_P_E (SPL3SMP_E) | 9 km | the same retrieval on Backus-Gilbert interpolated Tb |
| L4_SM (SPL4SMGP) | 9 km, 3-hourly | Tb assimilated into a land surface model: surface and root zone |
| L2_SM_SP (SPL2SMAP_S) | 3 and 1 km | SMAP Tb disaggregated with Sentinel-1 backscatter |

### AM and PM overpasses

SMAP is in a sun-synchronous orbit with a 6 am local descending (AM) and 6 pm ascending (PM)
equator crossing. The baseline retrieval uses the AM pass, when the soil, canopy and air
temperatures are closest, so the single effective temperature of the model is most valid; the L3
file has both, and the PM group's variables end in `_pm`. The swath is 1,000 km wide, so a location
is seen two or three times in three days.

### Quality flags

`retrieval_qual_flag` (bits: 0 recommended quality, 1 retrieval attempted, 2 retrieval successful,
3 freeze-thaw retrieval successful; a set bit means "not") and `surface_flag` (static and radar
water, coastal, urban, precipitation, snow, frozen ground, mountains, dense vegetation) describe
each retrieval. The recommended quality mask keeps retrievals whose bit 0 is clear: vegetation
water content below 5 kg m-2, little open water and no frozen soil, among other conditions.

### Climatology and anomalies

Soil moisture has a strong seasonal cycle. The day-of-year climatology (the mean for each calendar
day, smoothed over a 31-day window) is the baseline against which droughts and wet spells are
anomalies: `anomaly = theta - climatology(doy)`.
""")

nb.md("""
## SMAP L3 Enhanced with GeoLattice

| Function | Purpose |
|---|---|
| `sd.smap_l3e()` (wraps `sm.open_product("SMAP_L3_E", ...)`) | ten-year cube over the Midwest box, cached |
| `sm.decode_flags` | named bit layers of `retrieval_qual_flag` and `surface_flag` |
| `sm.mask_quality` | recommended, all or strict quality |
| `sm.daily_cube` | AM first, PM fill, regular daily calendar |
| `sm.sm_climatology`, `sm.sm_anomaly` | smoothed day-of-year climatology and anomalies |

### Learning objectives

1. Read SMAP L3 Enhanced retrievals with their ancillary layers for a region and ten years.
2. Decode the quality bits and see which conditions remove data in each season.
3. Build a daily cube from the AM and PM overpasses and measure the revisit.
4. Compute a smoothed day-of-year climatology and monthly climatology maps.
5. Map and plot the 2023 drought as anomalies.

### Data

| Dataset | Content |
|---|---|
| SMAP L3 Enhanced (SPL3SMP_E v006), 31 March 2015 to 31 December 2025 | 9 km AM and PM soil moisture, quality and surface flags, VWC, vegetation opacity, surface temperature and Tb; 101 W to 90 W, 40 N to 44 N |

### Contents

- Setup
- What is in the cube
- Quality flags through the year
- A daily cube from two overpasses
- Day-of-year climatology
- The 2023 drought as an anomaly
- Summary and exercises
""")

nb.setup('''
raw = sd.smap_l3e()  # cached SPL3SMP_E cube, AM and PM
iowa = sd.ease2_clip(raw, sd.IOWA_REGION)  # Iowa and its neighbours (97 W to 90 W)
print(raw)  # dimensions and variables
''')

nb.md("""
## What is in the cube
""")

nb.code('''
info = pd.DataFrame({v: {"dims": ", ".join(raw[v].dims), "dtype": str(raw[v].dtype),
                         "AM mean": float(raw[v].sel(overpass="AM").where(raw[v].sel(overpass="AM") < 65534).mean())}
                     for v in raw.data_vars}).T  # one row per variable
display(info)  # variables of the retrieval
print(f"{raw.sizes['time']} days, {raw.sizes['y']} x {raw.sizes['x']} cells of 9 km, "
      f"{raw.nbytes / 2**20:.0f} MB in memory")  # size
''')

nb.md("""
## Quality flags through the year
""")

nb.code('''
q = sm.decode_flags(iowa.retrieval_qual_flag.sel(overpass="AM"), "SMAP")  # retrieval bits
s = sm.decode_flags(iowa.surface_flag.sel(overpass="AM"), "SMAP_SURFACE")  # surface bits
attempted = ~q.retrieval_not_attempted & (iowa.retrieval_qual_flag.sel(overpass="AM") < 65534)  # cells in the swath with a retrieval attempt
month = iowa.time.dt.month  # calendar month
share = pd.DataFrame({
    "in swath, retrieval attempted": attempted.groupby(month).mean(...).values,
    "not recommended": q.not_recommended.where(attempted).groupby(month).mean(...).values,
    "frozen (radiometer)": s.frozen_radiometer.where(attempted).groupby(month).mean(...).values,
    "snow or ice": s.snow_ice.where(attempted).groupby(month).mean(...).values,
    "precipitation": s.precipitation.where(attempted).groupby(month).mean(...).values,
    "static water": s.static_water.where(attempted).groupby(month).mean(...).values,
}, index=range(1, 13)) * 100  # percent of AM cells (first column) and of attempted retrievals (others)
display(share.round(0))  # table
ax = share.plot(figsize=(10, 4), marker="o"); ax.set_xlabel("month"); ax.set_ylabel("%")  # plot
ax.set_title("SMAP L3E flags over Iowa, 2015 to 2025"); plt.tight_layout(); plt.show()  # draw
''')

nb.md("""
The AM swath covers about half of the region on a given day; in winter far fewer cells get a
retrieval attempt (11 % in January), because the retrieval is skipped over frozen ground and snow. Of
the attempted retrievals, 14 to 29 % are not of recommended quality depending on the month (mostly
cells with towns, lakes or, in winter, snow); the radiometer flags frozen ground in 18 % of January
retrievals and almost none in summer; precipitation at overpass time flags 2 to 5 %, static water 3 %.
""")

nb.md("""
## A daily cube from two overpasses
""")

nb.code('''
good = sm.mask_quality(iowa, level="recommended")  # recommended quality, no frozen or water cells
cube = sm.daily_cube(good, combine="am_first")  # AM first, PM where AM is missing
sm_ = cube.sm  # daily soil moisture
cover = sm_.notnull().mean(("y", "x"))  # fraction of the region with a retrieval each day
season = cover.where(cover.time.dt.month.isin([5, 6, 7, 8, 9]))  # growing season
gap = sm_.where(sm_.time.dt.month.isin([5, 6, 7, 8, 9])).notnull()  # observed or not, May to September
revisit = gap.sum("time") / gap.time.dt.year.to_series().nunique()  # observed days per season and cell
print(f"May to September: {float(season.mean()):.0%} of cells retrieved on an average day; "
      f"{float(revisit.median()):.0f} days of 153 per cell and season (median)")  # revisit
used = cube.overpass_used.where(cube.overpass_used >= 0)  # 0 AM, 1 PM
print(f"retrievals from the PM pass: {float((used == 1).sum() / used.notnull().sum()):.0%}")  # share filled by PM
fig, ax = plt.subplots(1, 2, figsize=(14, 4))  # coverage and one day
cover.sel(time="2024").plot(ax=ax[0]); ax[0].set_ylabel("share of cells"); ax[0].set_title("Daily coverage, 2024")  # coverage series
sm_.sel(time="2024-07-15").plot(ax=ax[1], vmin=0.05, vmax=0.45, cmap="YlGnBu"); ax[1].set_title("15 July 2024")  # one map
plt.tight_layout(); plt.show()  # draw
''')

nb.md("""
From May to September 51 % of the region has a recommended retrieval on an average day, and a
typical cell has one on 87 of the 153 days of the season; the evening pass contributes a quarter of
the cube. The map shows swath edges and the east-west soil moisture gradient that follows rainfall.
""")

nb.md("""
## Day-of-year climatology
""")

nb.code('''
clim = sm.sm_climatology(sm_, window=31, period=("2015-04-01", "2025-12-31"))  # smoothed day-of-year mean
mean_curve = clim.mean(("y", "x")).to_series()  # regional climatology
fig, ax = plt.subplots(1, 4, figsize=(18, 3.5), gridspec_kw={"width_ratios": [1.4, 1, 1, 1]})  # curve and three months
ax[0].plot(mean_curve.index, mean_curve.values); ax[0].set_xlabel("day of year"); ax[0].set_ylabel("m3/m3")  # seasonal cycle
ax[0].set_title("Regional mean climatology")  # title
for a, (doy, name) in zip(ax[1:], ((105, "mid April"), (196, "mid July"), (288, "mid October"))):  # three months
    clim.sel(doy=doy).plot(ax=a, vmin=0.1, vmax=0.4, cmap="YlGnBu"); a.set_title(f"Climatology, {name}"); a.axis("off")  # maps
plt.tight_layout(); plt.show()  # draw
print(f"wettest day of year {int(mean_curve.idxmax())} ({mean_curve.max():.3f}), driest {int(mean_curve.idxmin())} ({mean_curve.min():.3f})")  # extremes
clim.to_netcdf(OUT / "smap_l3e_climatology_iowa.nc")  # save for later notebooks
''')

nb.md("""
The regional mean peaks at about 0.26 m3 m-3 around day 80 (late March), after snowmelt and spring
rain, and reaches its minimum of about 0.19 at day 213 (early August), when the crops transpire most.
The east is wetter than the west, following the rainfall gradient across the Great Plains.
""")

nb.md("""
## The 2023 drought as an anomaly
""")

nb.code('''
anom = sm.sm_anomaly(sm_, clim=clim)  # theta minus the climatology of the same day
reg = pd.DataFrame({"soil moisture": sm_.mean(("y", "x")).to_series(), "climatology": None})  # regional means
doy = np.minimum(reg.index.dayofyear, 365)  # day of year
reg["climatology"] = mean_curve.reindex(doy).values  # regional climatology on each date
fig, ax = plt.subplots(1, 2, figsize=(15, 4), gridspec_kw={"width_ratios": [2, 1]})  # series and map
r = reg.loc["2022-04":"2024-10"].rolling(7, min_periods=3).mean()  # weekly smoothing
ax[0].plot(r.index, r["soil moisture"], label="SMAP, 7-day mean"); ax[0].plot(r.index, r["climatology"], "k--", label="climatology")  # series
ax[0].legend(); ax[0].set_ylabel("m3/m3"); ax[0].set_title("Iowa region mean, 2022 to 2024")  # labels
aug = anom.sel(time=slice("2023-08-01", "2023-08-31")).mean("time")  # August 2023 mean anomaly
aug.plot(ax=ax[1], cmap="BrBG", vmin=-0.12, vmax=0.12); ax[1].set_title("Anomaly, August 2023"); ax[1].axis("off")  # map
plt.tight_layout(); plt.show()  # draw
yearly = anom.where(anom.time.dt.month.isin([6, 7, 8])).groupby("time.year").mean(...).to_series()  # summer anomaly per year
display(yearly.rename("June to August anomaly (m3/m3)").to_frame().T.round(3))  # table
''')

nb.md("""
The table ranks the summers. 2023 had the most negative June to August anomaly of the decade
(-0.020 m3 m-3), followed by 2017, 2020 and 2021; 2015 and 2019 were the wettest. In 2023 soil
moisture stayed below the climatology from early summer to autumn across most of the region. Notebook 25 turns these anomalies into percentiles and
drought classes.

## Summary and exercises

- SMAP L3 Enhanced over the Midwest box is a cube of about 3,900 days of AM and PM retrievals with
  their flags and ancillary layers.
- Winter retrievals are few (frozen ground, snow); 14 to 29 % of attempted retrievals are not of
  recommended quality.
- With AM first and PM fill, about half of the region has a retrieval on a summer day, and a cell has
  one on 87 of 153 days of the season.
- The smoothed climatology peaks in late March (0.26 m3 m-3) and bottoms out in early August (0.19);
  2023 was the driest summer of the record.

**Exercises**

1. Use `combine="mean"` in `sm.daily_cube`. Where do the AM and PM retrievals differ most, and in
   which season?
2. Apply `level="strict"` in `sm.mask_quality`. How many more retrievals are removed, and is the 2023
   anomaly different?
3. Plot the vegetation water content (`vegetation_water_content`) climatology for July. Which
   parts of the region exceed 5 kg m-2?
4. Compute the climatology from 2015 to 2020 only and recompute the 2023 anomaly. How sensitive is
   an anomaly to the reference period?
""")

nb.write()
