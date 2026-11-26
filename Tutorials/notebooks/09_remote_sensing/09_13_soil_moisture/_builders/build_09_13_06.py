"""Build 09_13_06_smos_and_smap_over_a_shared_period.ipynb: ``python _builders/run.py 06``."""

from nb import Notebook

nb = Notebook("09_13_06_smos_and_smap_over_a_shared_period.ipynb")

nb.md("""
# SMOS and SMAP over a Shared Period

ESA's Soil Moisture and Ocean Salinity (SMOS) mission, launched in 2009, was the first L-band
radiometer built for soil moisture. Instead of a large rotating antenna it uses an interferometer
(MIRAS) of 69 small receivers on a Y-shaped structure, which synthesizes an image of each scene
at many incidence angles. SMAP and SMOS therefore observe the same physics with very different
instruments. NSIDC's SMOS-SMAP product (NSIDC-0800) retrieves soil moisture from SMOS brightness
temperatures with the SMAP algorithm and ancillary data, so the two records can be joined. This
notebook compares SMOS-SMAP with SMAP L3 Enhanced over Iowa and its neighbours from 2022 to 2024:
coverage, maps, per-cell agreement and its seasonal cycle.
""")

nb.md("""
## Overview

### Two L-band radiometers

| | SMAP | SMOS |
|---|---|---|
| Launch | January 2015 | November 2009 |
| Antenna | 6 m rotating mesh reflector | MIRAS interferometer, 69 receivers |
| Incidence | 40 degrees, constant | 0 to 60 degrees, multi-angular |
| Footprint | about 40 km | about 43 km (average) |
| Overpass (local) | 6 am descending, 6 pm ascending | 6 am ascending, 6 pm descending |
| RFI handling | hardware and software detection and filtering | strong RFI contamination in parts of Europe and Asia |

### Radio frequency interference

L-band 1.400 to 1.427 GHz is a protected band for radio astronomy and passive sensing, yet
illegal or out-of-band emitters (radars, TV links) leak into it. RFI adds warm brightness
temperature and makes soils look drier, or ruins retrievals altogether. SMOS suffered badly in its
early years over Europe and Asia; SMAP was designed with digital RFI filtering. Over the central
US both are mostly clean.

### Why a SMAP-algorithm SMOS product

ESA's own SMOS L2 and CATDS L3 products use a different retrieval (multi-angular, L-MEB model,
ECMWF ancillary data). Differences between SMOS and SMAP retrievals then mix instrument and
algorithm. NSIDC-0800 interpolates SMOS Tb to SMAP's 40 degrees and applies the SMAP baseline
retrieval, isolating the instrument difference and allowing a SMAP-consistent record that starts
in 2010.
""")

nb.md("""
## SMOS with GeoLattice

| Function | Purpose |
|---|---|
| `sd.smos()` (wraps `sm.open_product("SMOS_SMAP_L3", ...)`) | NSIDC-0800 cube 2022 to 2024, read by HTTP range requests |
| `sm.read_smos` | read a local NSIDC-0800 file, SMAP-based or SMOS-based group |
| `sm.mask_quality`, `sm.daily_cube` | same quality logic as SMAP |
| `sm.harmonize`, `sm.validation_metrics` | common grid and agreement statistics |

### Learning objectives

1. Read SMOS retrievals made with the SMAP algorithm and apply their quality flags.
2. Compare daily coverage of the two missions over the same region.
3. Map the per-cell correlation and unbiased differences of SMOS and SMAP.
4. Find in which seasons the two missions disagree and explain why.

### Data

| Dataset | Content |
|---|---|
| NSIDC-0800 v2 (SMOS-SMAP L3, 9 km), 2022 to 2024 | SMAP-algorithm soil moisture from SMOS Tb, AM and PM, Iowa region |
| SMAP L3 Enhanced (SPL3SMP_E v006), 2022 to 2024 | the SMAP reference on the same grid |

### Contents

- Setup
- Daily coverage of the two missions
- Maps of the period mean
- Per-cell agreement
- Agreement through the seasons
- Summary and exercises
""")

nb.setup('''
period = slice("2022-01-01", "2024-12-31")  # shared period
smos = sm.daily_cube(sm.mask_quality(sd.smos()))  # SMOS-SMAP, recommended quality, AM first
smap = sm.daily_cube(sm.mask_quality(sd.ease2_clip(sd.smap_l3e(), sd.IOWA_REGION).sel(time=period)))  # SMAP L3E
a, b = xr.align(smap.sm, smos.sm, join="inner")  # common cells and days (both are on the 9 km EASE grid)
both = xr.Dataset({"SMAP": a, "SMOS": b})  # one dataset
print(both.sizes)  # dimensions
''')

nb.md("""
## Daily coverage of the two missions
""")

nb.code('''
cov = both.notnull().mean(("y", "x")).to_dataframe()[["SMAP", "SMOS"]]  # share of cells retrieved per day
monthly = cov.groupby(cov.index.month).mean() * 100  # mean coverage by month
display(monthly.round(0).T)  # table
joint = (both.SMAP.notnull() & both.SMOS.notnull()).mean(("y", "x")).to_series()  # both on the same day
print(f"days with a retrieval from either: SMAP {float(cov.SMAP.mean()):.0%} of cells, SMOS {float(cov.SMOS.mean()):.0%}; "
      f"both on the same day {float(joint.mean()):.0%}")  # summary
ax = monthly.plot(marker="o", figsize=(9, 3.5)); ax.set_xlabel("month"); ax.set_ylabel("% of cells per day")  # plot
ax.set_title("Daily coverage, recommended quality, 2022 to 2024"); plt.tight_layout(); plt.show()  # draw
''')

nb.md("""
## Maps of the period mean
""")

nb.code('''
warm = both.where(both.time.dt.month.isin([5, 6, 7, 8, 9]))  # growing seasons only
fig, ax = plt.subplots(1, 3, figsize=(16, 3.6))  # SMAP, SMOS, difference
warm.SMAP.mean("time").plot(ax=ax[0], vmin=0.1, vmax=0.4, cmap="YlGnBu"); ax[0].set_title("SMAP, May to Sept mean")  # SMAP
warm.SMOS.mean("time").plot(ax=ax[1], vmin=0.1, vmax=0.4, cmap="YlGnBu"); ax[1].set_title("SMOS-SMAP, May to Sept mean")  # SMOS
(warm.SMOS - warm.SMAP).mean("time").plot(ax=ax[2], cmap="RdBu", vmin=-0.08, vmax=0.08); ax[2].set_title("SMOS minus SMAP")  # bias
for a in ax: a.axis("off")  # clean maps
plt.tight_layout(); plt.show()  # draw
''')

nb.md("""
## Per-cell agreement
""")

nb.code('''
pair = both.where(both.SMAP.notnull() & both.SMOS.notnull())  # same days only
r = xr.corr(pair.SMAP, pair.SMOS, "time")  # temporal correlation per cell
d = pair.SMOS - pair.SMAP  # differences
ubrmsd = np.sqrt((d ** 2).mean("time") - d.mean("time") ** 2)  # unbiased RMS difference
n = d.notnull().sum("time")  # paired days per cell
fig, ax = plt.subplots(1, 3, figsize=(16, 3.6))  # three maps
r.plot(ax=ax[0], vmin=0.3, vmax=1, cmap="viridis"); ax[0].set_title("Correlation")  # R
ubrmsd.plot(ax=ax[1], vmin=0, vmax=0.08, cmap="magma_r"); ax[1].set_title("ubRMSD (m3/m3)")  # ubRMSD
n.plot(ax=ax[2], cmap="Greys"); ax[2].set_title("Paired days")  # sample size
for a in ax: a.axis("off")  # clean maps
plt.tight_layout(); plt.show()  # draw
print(f"median R {float(r.median()):.2f}, median ubRMSD {float(ubrmsd.median()):.3f} m3/m3, median bias {float(d.mean('time').median()):+.3f}")  # summary
mean = pair.mean(("y", "x")).to_dataframe()[["SMAP", "SMOS"]].dropna()  # regional means on shared days
display(sm.validation_metrics(mean.SMAP, mean.SMOS, anomalies=True).to_frame("SMOS vs SMAP, regional mean").T.round(3))  # metrics
''')

nb.md("""
## Agreement through the seasons
""")

nb.code('''
rows = []  # metrics per month
dd = pair.to_dataframe()[["SMAP", "SMOS"]].dropna().reset_index()  # all paired cell-days
for m, g in dd.groupby(dd.time.dt.month):  # each calendar month
    met = sm.validation_metrics(g.SMAP, g.SMOS)  # SMOS against SMAP
    rows.append({"month": m, "n": int(met.n), "bias": met.bias, "ubrmsd": met.ubrmse, "R": met.pearson_r})  # row
seas = pd.DataFrame(rows).set_index("month")  # table
display(seas.round(3))  # metrics by month
fig, ax = plt.subplots(figsize=(13, 4))  # regional time series
mean.rolling(7, min_periods=3).mean().plot(ax=ax); ax.set_ylabel("m3/m3"); ax.set_title("Regional mean, 7-day smoothing")  # plot
plt.tight_layout(); plt.show()  # draw
''')

nb.md("""
The two missions agree closely in the regional mean, as expected for the same band, retrieval
and ancillary data. The per-cell differences have three origins: the different footprints and
gridding of the two radiometers, the interpolation of multi-angular SMOS Tb to 40 degrees, and the
different time of the overpasses relative to rain. The monthly table shows where the agreement is
weakest; spring and autumn transitions (frost, early crops) are the usual suspects.

## Summary and exercises

- SMOS-SMAP (NSIDC-0800) applies the SMAP algorithm to SMOS brightness temperatures, so differences
  with SMAP L3 Enhanced reflect mainly the instruments.
- Over Iowa both missions are almost free of RFI and give similar coverage; together they observe
  more cells per day than either alone.
- Per-cell correlations are high and unbiased differences are of the order of SMAP's accuracy
  target; the regional means nearly coincide.

**Exercises**

1. Re-read one day with `sm.read_smos(..., retrieval="SMOS_Based")` (download one NSIDC-0800 file
   with `sm.download_product`) and compare the two retrieval groups of the same file.
2. Merge SMAP and SMOS into one daily cube (SMAP first, SMOS where SMAP is missing) and recompute the
   daily coverage.
3. Repeat the per-cell agreement with the PM overpasses only (`combine="pm"`).
""")

nb.write()
