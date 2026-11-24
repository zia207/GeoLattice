"""County-level causal spatial data for the GeoLattice spatial causal tutorials (``geolattice.spatial_analysis.spatialcausal``).

Real data
---------
* County Health Rankings (CHR) 2024 county file and county polygons from the
  USA-Health Data Atlas repository (github.com/zia207/USA-Health-Data-Atlas).
* Contiguous U.S. (48 states + DC), projected to EPSG:5070 (Albers equal
  area); coordinates are kept in **kilometres**.

Semi-synthetic causal scenarios
-------------------------------
Real covariates, real geography and real 2024 adult diabetes prevalence are
kept. A *hypothetical* treatment and its effect are simulated on top, so every
estimator can be checked against a known truth while confounding patterns,
county shapes and neighbour structure stay realistic.

* :func:`scenario_program` — A: a county community-health-worker (CHW)
  diabetes program, assigned with spatial confounding, with heterogeneous
  effects and spillovers to neighbouring counties.
* :func:`scenario_state_policy` — B: a state-level policy whose map follows
  the 2024 Medicaid-expansion map (the assignment is real, the outcome effect
  is simulated), for the boundary regression discontinuity.
* :func:`scenario_panel_rings` (``post`` = calendar post-period for all units),
  :func:`scenario_panel_staggered` (``post`` = own adoption, treated units only),
  :func:`scenario_synthetic_control` — C: annual 2014–2025 panels.
* :func:`scenario_counts` — D: premature-death counts for Poisson models.

Nothing here is a real program evaluation: the effects are invented.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.special import expit

import geolattice

# The installed package lives in src/; the repository root is its parent.
ROOT = Path(geolattice.__file__).resolve().parents[1]
if not (ROOT / "Data").is_dir():
    raise FileNotFoundError("Could not find the GeoLattice repository root.")
# Only needed to rebuild the cache: a clone of github.com/zia207/USA-Health-Data-Atlas.
ATLAS = os.environ.get("ATLAS_REPO", str(ROOT.parent / "USA-Health-Data-Atlas"))
# GeoLattice keeps the cached county frame in Data/03_spatial_analysis/data_saptial_causal/ at the repository root.
DATA_DIR = os.environ.get(
    "SPATIAL_CAUSAL_DATA", str(ROOT / "Data" / "03_spatial_analysis" / "data_saptial_causal"))
CACHE = os.path.join(DATA_DIR, "county_frame.gpkg")

# States that had NOT adopted the ACA Medicaid expansion as of 2024-2025 (KFF: 10 states).
NON_EXPANSION = ["AL", "FL", "GA", "KS", "MS", "SC", "TN", "TX", "WI", "WY"]

COVARIATES = ["log_income", "uninsured_pct", "rural_pct", "age65_pct", "unemployment_pct",
              "child_poverty_pct", "smoking_pct", "inactivity_pct", "obesity_pct",
              "food_env", "log_pcp", "black_pct", "broadband_pct"]

LABELS = {
    "log_income": "log median household income", "uninsured_pct": "uninsured (% of population)",
    "rural_pct": "rural population (%)", "age65_pct": "aged 65+ (%)",
    "unemployment_pct": "unemployment (%)", "child_poverty_pct": "children in poverty (%)",
    "smoking_pct": "adult smoking (%)", "inactivity_pct": "physical inactivity (%)",
    "obesity_pct": "adult obesity (%)", "food_env": "food environment index (0-10)",
    "log_pcp": "log(1 + primary care physicians per 100k)", "black_pct": "Black population (%)",
    "broadband_pct": "households with broadband (%)", "diabetes_pct": "adult diabetes prevalence (%)",
}


# ----------------------------------------------------------------------------
# Real county frame
# ----------------------------------------------------------------------------
def _clean_chr(df: pd.DataFrame) -> pd.DataFrame:
    """Select and rename CHR 2024 measures.

    Several CHR columns in this file carry shifted labels (``unemployment_pct``
    holds a count while ``unemployed`` holds the percentage, ``rural`` is the
    percentage, ``black`` the percentage Black); the mapping below uses the
    columns whose values are percentages.
    """
    out = pd.DataFrame({
        "fips": df["fips"].astype(int),
        "state_abbr": df["state_abbr"],
        "county": df["county"],
        "population": df["population"],
        "diabetes_pct": df["diabetes_prevalence_adults_with_diabetes"],
        "log_income": np.log(df["median_household_income"]),
        "uninsured_pct": 100 * df["uninsured"] / df["population"],
        "rural_pct": df["rural"],
        "age65_pct": df["65_and_older_65_and_over"],
        "unemployment_pct": df["unemployed"],
        "child_poverty_pct": df["children_in_poverty"],
        "smoking_pct": df["adult_smoking_pct"],
        "inactivity_pct": df["physical_inactivity_pct"],
        "obesity_pct": df["adult_obesity_pct"],
        "food_env": df["food_environment_index"],
        "log_pcp": np.log1p(df["primary_care_physicians_rate"]),
        "black_pct": df["black"],
        "broadband_pct": df["broadband_access_pct"],
        "premature_deaths": df["premature_mortality_count"],
    })
    return out


def build_county_frame(force: bool = False) -> gpd.GeoDataFrame:
    """Contiguous-U.S. county GeoDataFrame (EPSG:5070) with clean CHR 2024 measures.

    Missing covariates are imputed with the state median, then the national
    median (about 5% of counties for the primary-care rate, under 2% for every
    other variable). Cached to ``Data/03_spatial_analysis/data_saptial_causal/county_frame.gpkg``
    in the GeoLattice repository (override with ``SPATIAL_CAUSAL_DATA``).
    """
    if os.path.exists(CACHE) and not force:
        return gpd.read_file(CACHE)
    g = gpd.read_file(os.path.join(ATLAS, "assets", "geojson_counties_fips.json"))
    g["fips"] = (g["STATE"] + g["COUNTY"]).astype(int)
    g.loc[g["fips"] == 46113, "fips"] = 46102  # Shannon -> Oglala Lakota County, SD (renamed 2015)
    chr24 = pd.read_csv(os.path.join(ATLAS, "data", "county_health_ranking", "chr_county_2024.csv"),
                        low_memory=False)
    df = _clean_chr(chr24)
    gdf = g[["fips", "geometry"]].merge(df, on="fips")
    gdf = gdf[~gdf["state_abbr"].isin(["AK", "HI", "PR"])].to_crs(5070)
    gdf = gdf.sort_values("fips").reset_index(drop=True)
    num = COVARIATES + ["diabetes_pct", "premature_deaths", "population"]
    gdf["n_imputed"] = gdf[num].isna().sum(1)
    for c in num:
        gdf[c] = gdf[c].fillna(gdf.groupby("state_abbr")[c].transform("median")).fillna(gdf[c].median())
    cent = gdf.geometry.centroid
    gdf["x_km"], gdf["y_km"] = cent.x / 1000.0, cent.y / 1000.0
    os.makedirs(DATA_DIR, exist_ok=True)
    gdf.to_file(CACHE, driver="GPKG")
    return gdf


@lru_cache(maxsize=2)
def queen_weights():
    """Queen contiguity weights (``geolattice.spatial_statistics.weights.Queen``) for the county frame."""
    from geolattice.spatial_statistics.weights import Queen
    return Queen(build_county_frame())


def coords_km(gdf) -> np.ndarray:
    return gdf[["x_km", "y_km"]].to_numpy()


def z(x) -> np.ndarray:
    x = np.asarray(x, float)
    return (x - x.mean()) / x.std()


def smooth_field(coords: np.ndarray, length_km: float, seed: int, n_bumps: int = 150) -> np.ndarray:
    """Standardised smooth random surface: a random sum of Gaussian bumps."""
    rng = np.random.default_rng(seed)
    lo, hi = coords.min(0), coords.max(0)
    centres = rng.uniform(lo, hi, size=(n_bumps, 2))
    amps = rng.normal(size=n_bumps)
    d2 = ((coords[:, None, :] - centres[None]) ** 2).sum(-1)
    f = (np.exp(-0.5 * d2 / length_km ** 2) * amps).sum(1)
    return (f - f.mean()) / f.std()


def _county_xy(gdf, state, county):
    r = gdf[(gdf.state_abbr == state) & (gdf.county == county)]
    return r[["x_km", "y_km"]].to_numpy()[0]


# ----------------------------------------------------------------------------
# Scenario A: county CHW program
# ----------------------------------------------------------------------------
def scenario_program(seed: int = 2024):
    """County community-health-worker (CHW) diabetes program (hypothetical).

    Assignment (logit):
        0.5 z(child poverty) + 0.4 z(uninsured) - 0.3 z(rural) + 0.3 z(log PCP) + 1.0 U
    where ``U`` is an *unmeasured*, spatially smooth "regional health-policy
    climate" (length scale 350 km), which also raises diabetes prevalence
    (+0.9 points per SD) — a spatially smooth unmeasured confounder.

    Effect on diabetes prevalence (percentage points):
        tau_i = -0.8 - 0.35 z(child poverty)_i - 0.6 * bump(distance to Jackson, MS; 400 km)
    so the program helps more in poorer counties and in the Deep South.
    Spillover: delta * (share of treated Queen neighbours), delta = -0.4.

    Columns added: ``U``, ``e_true``, ``T``, ``tau``, ``share_treated``,
    ``y0`` (diabetes without treatment), ``y_noint`` (no spillover) and ``y``.
    """
    from geolattice.spatial_analysis.spatialcausal._utils import as_sparse_weights

    gdf = build_county_frame().copy()
    w = queen_weights()
    xy = coords_km(gdf)
    rng = np.random.default_rng(seed)
    U = smooth_field(xy, 350.0, seed)
    lin = (-0.55 + 0.5 * z(gdf.child_poverty_pct) + 0.4 * z(gdf.uninsured_pct) - 0.3 * z(gdf.rural_pct)
           + 0.3 * z(gdf.log_pcp) + 1.0 * U)
    e = expit(lin)
    T = rng.binomial(1, e)
    jackson = _county_xy(gdf, "MS", "Hinds")
    bump = np.exp(-0.5 * ((xy - jackson) ** 2).sum(1) / 400.0 ** 2)
    tau = -0.8 - 0.35 * z(gdf.child_poverty_pct) - 0.6 * bump
    Wr = as_sparse_weights(w, len(gdf), row_standardize=True)
    share = np.asarray(Wr @ T).ravel()
    delta = -0.4
    y0 = gdf.diabetes_pct.to_numpy() + 0.9 * U
    gdf["U"], gdf["e_true"], gdf["T"], gdf["tau"], gdf["share_treated"] = U, e, T, tau, share
    gdf["y0"] = y0
    gdf["y_noint"] = y0 + tau * T
    gdf["y"] = y0 + tau * T + delta * share
    ov = e * (1 - e)
    truth = {"ate": float(tau.mean()), "att": float(tau[T == 1].mean()), "delta": delta,
             "plm_estimand": float(np.sum(ov * tau) / np.sum(ov)),
             "total_all_treated": float(tau.mean() + delta)}
    return gdf, w, truth


# ----------------------------------------------------------------------------
# Scenario B: state policy and boundary RDD
# ----------------------------------------------------------------------------
def scenario_state_policy(effect: float = -0.6, heterogeneity: float = 0.3, seed: int = 7):
    """State policy following the 2024 Medicaid-expansion map; simulated effect on diabetes.

    ``T_state = 1`` in expansion states. The simulated jump at the border is
    ``effect + heterogeneity * cos(2 pi * position along the border)``.
    ``y_rdd = y0 + jump * T_state`` with ``y0`` the real 2024 diabetes
    prevalence plus the smooth confounder ``U`` (continuous across borders).

    Returns ``(gdf, boundary, truth)`` where ``boundary`` is the shapely
    (Multi)LineString of land borders between expansion and non-expansion
    states (coastlines excluded).
    """
    import shapely

    gdf = build_county_frame().copy()
    xy = coords_km(gdf)
    gdf["T_state"] = (~gdf.state_abbr.isin(NON_EXPANSION)).astype(int)
    states = gdf.dissolve("state_abbr")[["geometry", "T_state"]].copy()
    states["geometry"] = states.buffer(0)
    treated = shapely.union_all(states[states.T_state == 1].geometry.values)
    control = shapely.union_all(states[states.T_state == 0].geometry.values)
    border = shapely.intersection(treated.buffer(200), control.buffer(200))      # metres
    border = shapely.intersection(shapely.boundary(control), border)
    border = shapely.line_merge(border) if border.geom_type != "LineString" else border
    from shapely.ops import transform
    boundary = transform(lambda x, y, z_=None: (np.asarray(x) / 1000.0, np.asarray(y) / 1000.0), border)
    U = smooth_field(xy, 350.0, 2024)
    gdf["U"] = U
    gdf["y0"] = gdf.diabetes_pct.to_numpy() + 0.9 * U
    pos = np.asarray([boundary.project(shapely.Point(p), normalized=True) for p in xy])
    jump = effect + heterogeneity * np.cos(2 * np.pi * pos)
    gdf["border_pos"] = pos
    gdf["y_rdd"] = gdf["y0"] + jump * gdf["T_state"]
    truth = {"effect": effect, "heterogeneity": heterogeneity,
             "jump_at": lambda p: effect + heterogeneity * np.cos(2 * np.pi * np.asarray(p))}
    return gdf, boundary, truth


# ----------------------------------------------------------------------------
# Scenario C: annual panels 2014-2025
# ----------------------------------------------------------------------------
YEARS = np.arange(2014, 2026)


def _panel_base(gdf, seed, spatial_loadings: bool = False):
    """y0_it = alpha_i + lambda_t + L_i' F_t + e_it.

    Loadings ``L`` are standardised covariates (child poverty, rural, 65+).
    With ``spatial_loadings=True`` two smooth spatial fields (120 km) are
    added, so nearby counties share trajectories — realistic, and what makes
    neighbours the most attractive synthetic-control donors.
    """
    rng = np.random.default_rng(seed)
    n, T = len(gdf), len(YEARS)
    alpha = gdf.diabetes_pct.to_numpy()
    lam = 0.08 * np.arange(T) + rng.normal(0, 0.03, T).cumsum()
    cols = [z(gdf.child_poverty_pct), z(gdf.rural_pct), z(gdf.age65_pct)]
    if spatial_loadings:
        xy = coords_km(gdf)
        cols += [1.5 * smooth_field(xy, 120.0, seed + 1), 1.5 * smooth_field(xy, 120.0, seed + 2)]
    L = np.column_stack(cols)
    F = rng.normal(0, 0.12, size=(T, L.shape[1])).cumsum(0)
    return alpha[:, None] + lam[None, :] + L @ F.T + rng.normal(0, 0.2, size=(n, T))


def _to_panel(gdf, Y, treated, post_matrix, truth):
    from geolattice.spatial_analysis.spatialcausal import SpatialPanel
    n, T = Y.shape
    unit = np.repeat(gdf.fips.to_numpy(), T)
    year = np.tile(YEARS, n)
    return SpatialPanel(unit, year, Y.ravel(), gdf.fips.to_numpy(), coords_km(gdf), treated=treated,
                        post=post_matrix.ravel(), truth=truth)


def scenario_panel_rings(adopt_year: int = 2019, n_states: int = 10, seed: int = 11):
    """Single adoption date: ``n_states`` random states adopt in ``adopt_year``.

    Direct effect -0.5 points. Untreated counties near treated ones receive a
    distance-decaying spillover: -0.4 (0-40 km), -0.2 (40-80 km),
    -0.1 (80-120 km), 0 beyond (distance between county centroids to the
    nearest treated county).
    """
    gdf = build_county_frame()
    rng = np.random.default_rng(seed)
    states = np.array(sorted(gdf.state_abbr.unique()))
    chosen = rng.choice(states, n_states, replace=False)
    treated = gdf.state_abbr.isin(chosen).to_numpy().astype(int)
    xy = coords_km(gdf)
    d, _ = cKDTree(xy[treated == 1]).query(xy)
    rings, spill_vals = (0, 40, 80, 120), (-0.4, -0.2, -0.1)
    spill = np.zeros(len(gdf))
    for r in range(3):
        m = (treated == 0) & (d > rings[r]) & (d <= rings[r + 1])
        spill[m] = spill_vals[r]
    Y = _panel_base(gdf, seed)
    post = (YEARS >= adopt_year)[None, :].repeat(len(gdf), 0)
    Y = Y + post * (-0.5 * treated[:, None] + spill[:, None])
    truth = {"att": -0.5, "rings_km": rings, "spillovers": spill_vals, "states": list(chosen),
             "adopt_year": adopt_year}
    # SpatialDiD reads ``post`` as a calendar indicator for every unit (rings switch on too)
    return _to_panel(gdf, Y, treated, post.astype(int), truth), gdf


def scenario_panel_staggered(cohorts=(2017, 2019, 2021), states_per_cohort: int = 6,
                             exposure_km: float = 60.0, seed: int = 21):
    """Staggered state adoption with spillovers to nearby never-treated counties.

    Direct effect ``k`` years after adoption: ``-0.4 - 0.1 k``. Never-treated
    counties within ``exposure_km`` of an adopting county get -0.3 from the
    year they are first exposed.
    """
    gdf = build_county_frame()
    rng = np.random.default_rng(seed)
    states = list(rng.permutation(sorted(gdf.state_abbr.unique())))
    adopt = np.full(len(gdf), np.inf)
    for c, yr in enumerate(cohorts):
        for s in states[c * states_per_cohort:(c + 1) * states_per_cohort]:
            adopt[(gdf.state_abbr == s).to_numpy()] = yr
    xy = coords_km(gdf)
    expo = np.full(len(gdf), np.inf)
    for yr in YEARS:
        src = np.flatnonzero(adopt <= yr)
        if len(src):
            hits = cKDTree(xy[src]).query_ball_point(xy, r=exposure_km)
            for i, h in enumerate(hits):
                if np.isinf(expo[i]) and any(src[j] != i for j in h):
                    expo[i] = yr
    Y = _panel_base(gdf, seed)
    k = YEARS[None, :] - adopt[:, None]
    direct = np.where(k >= 0, -0.4 - 0.1 * k, 0.0)
    spill = np.where(np.isinf(adopt)[:, None] & (YEARS[None, :] >= expo[:, None]), -0.3, 0.0)
    Y = Y + direct + spill
    treated = np.isfinite(adopt).astype(int)
    post = (YEARS[None, :] >= adopt[:, None]).astype(int)
    truth = {"event_time": lambda kk: -0.4 - 0.1 * np.asarray(kk), "spillover": -0.3,
             "exposure_km": exposure_km, "cohorts": {yr: states[c * states_per_cohort:(c + 1) * states_per_cohort]
                                                     for c, yr in enumerate(cohorts)}}
    return _to_panel(gdf, Y, treated, post, truth), gdf


def scenario_synthetic_control(state_list=("MS", "AL", "LA"), treated_county=("MS", "Hinds"),
                               adopt_year: int = 2021, effect: float = -1.0, spill: float = -0.8,
                               spill_km: float = 120.0, seed: int = 31):
    """One treated county in a regional panel; nearby counties receive spillover.

    Spillover ``spill * (1 - d / spill_km)`` for counties within ``spill_km``
    of the treated county. Donors are limited to ``state_list`` so the
    synthetic control and its placebos stay fast.
    """
    gdf_all = build_county_frame()
    Y_all = _panel_base(gdf_all, seed, spatial_loadings=True)
    keep = gdf_all.state_abbr.isin(state_list).to_numpy()
    gdf = gdf_all[keep].reset_index(drop=True)
    Y = Y_all[keep]
    xy = coords_km(gdf)
    tr = int(np.flatnonzero((gdf.state_abbr == treated_county[0]) & (gdf.county == treated_county[1]))[0])
    d = np.sqrt(((xy - xy[tr]) ** 2).sum(1))
    post = YEARS >= adopt_year
    sp = np.where((d > 0) & (d < spill_km), spill * (1 - d / spill_km), 0.0)
    eff = sp.copy()
    eff[tr] = effect
    Y = Y + eff[:, None] * post[None, :]
    treated = np.zeros(len(gdf), int)
    treated[tr] = 1
    postm = (post[None, :] & (treated[:, None] == 1)).astype(int)
    truth = {"att": effect, "spillover_by_unit": sp, "spill_km": spill_km,
             "treated_fips": int(gdf.fips[tr]), "adopt_year": adopt_year}
    return _to_panel(gdf, Y, treated, postm, truth), gdf


# ----------------------------------------------------------------------------
# Scenario D: counts
# ----------------------------------------------------------------------------
def scenario_counts(states=None, effect: float = -0.10, delta: float = -0.06, seed: int = 41):
    """Premature-death counts with a simulated program effect on the log rate.

    ``E_i`` = population x national premature-death rate (expected counts).
    ``O_i ~ Poisson(E_i * exp(log SMR_i + effect * T_i + delta * share_i + 0.15 U_i))``
    where ``SMR_i`` is the real CHR 2024 premature-death ratio, and ``T``,
    ``U``, ``share`` come from scenario A. Effects are log rate ratios.
    Optionally restrict to ``states``.
    """
    from geolattice.spatial_analysis.spatialcausal._utils import as_sparse_weights
    from geolattice.spatial_statistics.weights import Queen

    gdf, _, _ = scenario_program()
    if states is not None:
        gdf = gdf[gdf.state_abbr.isin(states)].reset_index(drop=True)
    w = Queen(gdf)
    rng = np.random.default_rng(seed)
    rate = gdf.premature_deaths.sum() / gdf.population.sum()
    E = gdf.population.to_numpy() * rate
    smr = np.clip(gdf.premature_deaths.to_numpy() / E, 0.2, 5)
    share = np.asarray(as_sparse_weights(w, len(gdf), row_standardize=True) @ gdf["T"].to_numpy()).ravel()
    lam = E * np.exp(np.log(smr) + effect * gdf["T"].to_numpy() + delta * share + 0.15 * gdf["U"].to_numpy())
    gdf["expected"] = E
    gdf["share_treated"] = share
    gdf["deaths_sim"] = rng.poisson(lam)
    return gdf, w, {"direct": effect, "spillover": delta, "total": effect + delta}


# ----------------------------------------------------------------------------
# Exposure-mapping estimands under a known assignment mechanism
# ----------------------------------------------------------------------------
def implied_exposure_contrasts(prob, w, tau, delta, threshold: float = 0.5, n_sims: int = 2000, seed: int = 0):
    """True exposure-condition contrasts when spillovers act through the continuous share.

    With ``Y_i = Y_i(0) + tau_i T_i + delta (WT)_i`` and a binary exposure mapping
    (own treatment x share >= threshold), the estimand of an inverse-probability or
    AIPW estimator of mu(c) is the county average of ``E[Y_i | C_i = c]``, which
    involves each county's *expected* share given its condition. Treatment is drawn
    ``n_sims`` times from the assignment probabilities ``prob`` (true propensities, or
    the design probability) to compute ``E[(WT)_i | C_i = c]``.

    Returns ``(contrasts, pi)`` with ``pi[i, c] = P(C_i = c)``.
    """
    from geolattice.spatial_analysis.spatialcausal._utils import as_sparse_weights

    n = len(tau)
    Wr = as_sparse_weights(w, n, row_standardize=True)
    rng = np.random.default_rng(seed)
    p = np.broadcast_to(np.asarray(prob, float), (n,))
    s_sum, cnt = np.zeros((n, 4)), np.zeros((n, 4))
    rows = np.arange(n)
    for _ in range(n_sims):
        t = (rng.random(n) < p).astype(int)
        sh = np.asarray(Wr @ t).ravel()
        c = 2 * t + (sh >= threshold)
        s_sum[rows, c] += sh
        cnt[rows, c] += 1
    with np.errstate(invalid="ignore"):
        es = s_sum / cnt                        # E[share | C = c] per county
    ok = lambda a, b: (cnt[:, a] > 0) & (cnt[:, b] > 0)  # noqa: E731
    def contrast(a, b, own):
        m = ok(a, b)
        return float(np.mean(own[m] + delta * (es[m, a] - es[m, b])))
    zero = np.zeros(n)
    out = {"direct": contrast(2, 0, tau), "spillover": contrast(1, 0, zero),
           "spillover_treated": contrast(3, 2, zero), "total": contrast(3, 0, tau)}
    return out, cnt / n_sims
