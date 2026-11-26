"""Build 09_13_05_smap_l4_surface_and_root_zone_soil_moisture.ipynb: ``python _builders/run.py 05``."""

from nb import Notebook

nb = Notebook("09_13_05_smap_l4_surface_and_root_zone_soil_moisture.ipynb")

nb.md("""
# SMAP L4: Surface and Root-Zone Soil Moisture

A radiometer sees the top 5 cm of soil every two to three days. Agriculture and hydrology need
the top metre, every day. The SMAP Level-4 soil moisture product (SPL4SMGP) bridges the gap by
assimilating SMAP brightness temperatures into the NASA Catchment land surface model with an
ensemble Kalman filter, driven by observed precipitation. It delivers gap-free surface (0 to 5 cm)
and root-zone (0 to 100 cm) soil moisture every three hours at 9 km. This notebook compares the
two layers over the Midwest, follows them through storms at the USCRN Des Moines 17 E station, and
checks both against the station's sensors.
""")

nb.md("""
## Overview

### Data assimilation in one paragraph

The land model integrates rain, evaporation, drainage and runoff to predict soil moisture at all
depths; the satellite observes the surface. The ensemble Kalman filter runs the model many times
with perturbed inputs, compares the predicted Tb with SMAP's, and nudges the state in proportion
to the ratio of model and observation uncertainty. Because model layers are coupled, a correction
of the surface also moves the root zone. The result keeps the physical consistency of the model
and the information of the observations (Reichle et al. 2019).

### What L4 delivers

| Variable | Layer | Notes |
|---|---|---|
| `sm_surface` | 0 to 5 cm | comparable with SMAP L3 |
| `sm_rootzone` | 0 to 100 cm | the agronomic layer |
| `sm_profile` | 0 cm to bedrock | full column |
| plus soil temperature, snow, fluxes | | |

Every 3 hours (01:30, 04:30, ... 22:30 UTC) at 9 km on the EASE-Grid 2.0, with a latency of a few
days. The chapter cache keeps the 10:30 UTC analysis of each day (04:30 local standard time in
Iowa, close to SMAP's morning overpass).

### Surface and root zone dynamics

After rain the surface wets almost at once and dries within days by evaporation and drainage.
The root zone integrates many events, responds later and less, and dries slowly by transpiration
through the season. The root zone is therefore a low-pass filtered, delayed version of the
surface, which notebook 24 models directly with an exponential filter.
""")

nb.md("""
## SMAP L4 with GeoLattice

| Function | Purpose |
|---|---|
| `sd.smap_l4()` (wraps `sm.open_product("SMAP_L4", ...)`) | daily 10:30 UTC analyses, 2015 to 2025, subset by NASA Harmony |
| `sm.read_smap` | read a full or subset L4 granule |
| `sm.sm_climatology`, `sm.sm_anomaly` | seasonal cycle and anomalies |
| `sm.validation_metrics` | bias, RMSE, ubRMSE and correlation against in situ data |

### Learning objectives

1. Read SMAP L4 surface and root-zone soil moisture for a region.
2. Compare the spatial patterns and variability of the two layers.
3. Follow both layers through storms at a cropland station and measure the delay of the root zone.
4. Validate surface and root-zone soil moisture against multi-depth in situ data.

### Data

| Dataset | Content |
|---|---|
| SMAP L4 (SPL4SMGP v008), 10:30 UTC each day, 2015 to 2025 | surface and root-zone soil moisture at 9 km, 101 W to 90 W, 40 N to 44 N |
| USCRN Des Moines 17 E | hourly soil moisture at 5 to 100 cm and precipitation |
| SMAP L3 Enhanced | AM retrievals at the station cell, for comparison |

### Contents

- Setup
- Surface and root zone over the Midwest
- Variability of the two layers
- Storms at a cropland station
- How late does the root zone respond
- Validation against the station
- Summary and exercises
""")

nb.setup('''
l4 = sd.smap_l4()  # daily SMAP L4 analyses (10:30 UTC)
l4["time"] = l4.time.dt.floor("D")  # one value per day
st = sd.STATIONS.loc["USCRN Des Moines 17 E"]  # station
r, c = sm.ease2_index(st.lat, st.lon, "EASE2_M09km")  # station cell
grid = sm.ease2_grid("EASE2_M09km", aoi=sd.MIDWEST)  # cube grid
px = l4.isel(x=int(np.where(grid.col == c)[0][0]), y=int(np.where(grid.row == r)[0][0]))  # station pixel
u = sd.uscrn()  # hourly USCRN record
ud = u.resample("1D").mean(); ud["p_mm"] = u.p_mm.resample("1D").sum(min_count=1)  # daily means and rain
ud["rootzone"] = ud[["sm_5", "sm_10", "sm_20", "sm_50", "sm_100"]].mean(axis=1, skipna=False)  # simple 0 to 100 cm mean
print(l4.sizes, "| station cell row", int(r), "col", int(c))  # dimensions
''')

nb.md("""
## Surface and root zone over the Midwest
""")

nb.code('''
days = ["2023-05-15", "2023-08-20"]  # wet spring and dry late summer
fig, ax = plt.subplots(2, 2, figsize=(14, 6))  # two layers, two days
for j, d in enumerate(days):  # each day
    for i, v in enumerate(("sm_surface", "sm_rootzone")):  # each layer
        l4[v].sel(time=d).plot(ax=ax[i, j], vmin=0.05, vmax=0.45, cmap="YlGnBu", cbar_kwargs={"label": "m3/m3"})  # map
        ax[i, j].set_title(f"{v.replace('sm_', '')}, {d}"); ax[i, j].axis("off")  # title
plt.tight_layout(); plt.show()  # draw
''')

nb.md("""
In spring both layers are wet; by late summer the surface has dried everywhere while the root zone
keeps a stronger east-west contrast, reflecting the season's accumulated rainfall and soil storage.
""")

nb.md("""
## Variability of the two layers
""")

nb.code('''
stats = pd.DataFrame({v: {"temporal std (median of cells)": float(l4[v].std("time").median()),
                          "day-to-day change, median abs": float(abs(l4[v].diff("time")).median()),
                          "lag-1 autocorrelation (station)": px[v].to_series().autocorr(1),
                          "lag-10 autocorrelation (station)": px[v].to_series().autocorr(10)} for v in ("sm_surface", "sm_rootzone")}).T  # statistics
display(stats.round(3))  # table
''')

nb.md("""
The root zone varies less in time and changes much more slowly from day to day: its ten-day
autocorrelation is close to one. This persistence is what makes root-zone soil moisture useful for
forecasting crop stress and runoff.
""")

nb.md("""
## Storms at a cropland station
""")

nb.code('''
sel = slice("2024-04-01", "2024-10-31")  # growing season 2024
df = pd.DataFrame({"L4 surface": px.sm_surface.to_series(), "L4 root zone": px.sm_rootzone.to_series()}).loc[sel]  # L4 at the station
fig, ax = plt.subplots(2, 1, figsize=(13, 6), sharex=True, gridspec_kw={"height_ratios": [1, 2.5]})  # rain and soil
ax[0].bar(ud.loc[sel].index, ud.loc[sel].p_mm, width=1); ax[0].set_ylabel("rain (mm)")  # station rain
df.plot(ax=ax[1], lw=1.4)  # L4 layers
ud.loc[sel, "sm_5"].plot(ax=ax[1], color="tab:blue", ls=":", label="USCRN 5 cm")  # in situ surface
ud.loc[sel, "rootzone"].plot(ax=ax[1], color="tab:orange", ls=":", label="USCRN 5 to 100 cm mean")  # in situ root zone
ax[1].legend(ncol=2); ax[1].set_ylabel("m3/m3"); ax[0].set_title("SMAP L4 and USCRN at Des Moines 17 E, 2024"); plt.tight_layout(); plt.show()  # draw
''')

nb.md("""
L4 follows the timing of the storms closely, because the model is driven by observed
precipitation, and the root zone responds to the large events but ignores the small ones. The
level of the in situ root zone is higher than L4's here, a site-specific offset that comes from the
model's soil parameters for the 9 km cell; the dynamics are what L4 captures best.
""")

nb.md("""
## How late does the root zone respond

The cross-correlation of daily rain with the day-to-day increase of each layer shows when the
response peaks.
""")

nb.code('''
p = ud.p_mm.loc["2015-04":"2025-10"]  # daily station rain
rows = []  # correlation by lag
for v in ("sm_surface", "sm_rootzone"):  # each L4 layer
    dv = px[v].to_series().diff()  # daily change
    for lag in range(0, 8):  # rain leads by 0 to 7 days
        rows.append({"layer": v, "lag (days)": lag, "r": p.corr(dv.shift(-lag))})  # correlation
xc = pd.DataFrame(rows).pivot(index="lag (days)", columns="layer", values="r")  # table
display(xc.round(2))  # correlations
''')

nb.md("""
Because the L4 file is the 10:30 UTC analysis, rain that falls during the day appears in the
next day's value, so the strongest correlation is at lag 1 for both layers. The surface signal is
stronger, but the root zone carries the response for longer.
""")

nb.md("""
## Validation against the station
""")

nb.code('''
pairs = {"L4 surface vs USCRN 5 cm": (ud.sm_5, px.sm_surface.to_series()),
         "L4 root zone vs USCRN 5-100 cm": (ud.rootzone, px.sm_rootzone.to_series())}  # comparisons
l3e = sd.smap_l3e()  # SMAP L3 Enhanced at the same cell
g9 = sm.ease2_grid("EASE2_M09km", aoi=sd.MIDWEST)  # grid of the L3E cube
l3px = l3e.sm.sel(overpass="AM").isel(x=int(np.where(g9.col == c)[0][0]), y=int(np.where(g9.row == r)[0][0])).to_series()  # AM retrievals
pairs["L3E AM vs USCRN 5 cm"] = (ud.sm_5, l3px)  # retrieval for comparison
warm = lambda s: s[(s.index.month >= 4) & (s.index.month <= 10)]  # unfrozen months
tab = pd.DataFrame({k: sm.validation_metrics(warm(a), warm(b), anomalies=True) for k, (a, b) in pairs.items()}).T  # metrics
display(tab[["n", "bias", "rmse", "ubrmse", "pearson_r", "anomaly_r"]].astype(float).round(3))  # table
''')

nb.md("""
After removing the bias (ubRMSE), L4 surface soil moisture agrees with the 5 cm sensor about as
well as the L3 retrieval, and it has a value every day instead of every two to three days. The
root-zone comparison uses a simple mean of five sensors, which is not the same layer average as the
model's, so its bias says more about definitions than about skill; its correlation and anomaly
correlation show that L4 tracks the wetting and drying of the profile.

## Summary and exercises

- SMAP L4 assimilates SMAP Tb into a land model and gives daily, gap-free surface and root-zone
  soil moisture at 9 km.
- The root zone varies less and more slowly than the surface; its ten-day autocorrelation is near one.
- Storms show in both layers within a day of the rain (in the 10:30 UTC analysis); the root zone
  holds the response for weeks.
- At the USCRN station L4 surface soil moisture has an anomaly correlation and ubRMSE comparable
  with the L3 retrieval, with no gaps.

**Exercises**

1. Repeat the validation at SCAN Ames. Is the root-zone bias of the same sign?
2. Compute the cross-correlation of rain with the USCRN 5 cm and 50 cm changes. Is the in situ
   delay longer than L4's?
3. Load the 01:30 and 13:30 UTC analyses with `sd.smap_l4(hours=("013000", "133000"))` for one month
   and measure the diurnal cycle of the surface layer.
""")

nb.write()
