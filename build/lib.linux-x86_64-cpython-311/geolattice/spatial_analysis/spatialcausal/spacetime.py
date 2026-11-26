"""Space-time causal inference: staggered adoption with spatial spillovers.

:class:`StaggeredSpatialDiD` extends the group-time ATT estimator of
Callaway & Sant'Anna (2021) to settings where treatment spills over in space.

* Every unit has an adoption time ``g`` (∞ for never treated) and an
  *exposure time* ``e``: the first period in which some unit within
  ``exposure_distance`` of it has been treated.
* **Direct effects** ATT(g, t) compare cohort ``g`` with *clean* controls:
  units neither treated nor exposed by period ``max(t, g - 1)``. Exposed
  neighbours are therefore never used as controls, which is the bias that
  :class:`~geolattice.spatial_analysis.spatialcausal.quasi.SpatialDiD` removes with rings in the
  single-date case.
* **Spillover effects** ATT_s(e, t) treat never-treated units first exposed
  at ``e`` as their own cohort, against the same clean controls.
* All comparisons use the universal base period (the period before the
  cohort starts), so pre-period estimates are placebo tests.
* Cohort–time cells are aggregated into an overall ATT and an event study.
  Inference is a cluster bootstrap over units or spatial blocks.

Static TWFE on staggered data is biased whenever effects change over time
(Goodman-Bacon, 2021); ``naive_twfe_`` is reported for comparison.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree

from geolattice.spatial_analysis.spatialcausal.core.panel import SpatialPanel
from geolattice.spatial_analysis.spatialcausal.core.result import EffectEstimate, SpatialCausalResult
from geolattice.spatial_analysis.spatialcausal.crossfit import SpatialBlockKFold


class StaggeredSpatialDiD:
    """Group-time DiD for staggered adoption with spatial spillovers.

    Parameters
    ----------
    exposure_distance : float
        A unit counts as exposed once any treated unit lies within this
        distance. Choose it beyond the plausible spillover range: units
        that are affected but not flagged as exposed contaminate controls.
    control : {"not_yet", "never"}
        Clean controls are not-yet-treated (and not-yet-exposed) units, or
        never-treated and never-exposed-by-then units only.
    min_cohort : int
        Cohorts smaller than this are dropped from aggregation.
    cluster : {"unit", "spatial"}
        Bootstrap resamples units, or spatial blocks of units.
    n_boot : int
        Bootstrap replications.
    """

    def __init__(self, exposure_distance: float, control: str = "not_yet", min_cohort: int = 5,
                 cluster: str = "unit", n_clusters="auto", n_boot: int = 199, random_state: int = 0):
        if control not in ("not_yet", "never"):
            raise ValueError("control must be 'not_yet' or 'never'")
        if cluster not in ("unit", "spatial"):
            raise ValueError("cluster must be 'unit' or 'spatial'")
        self.exposure_distance = float(exposure_distance)
        self.control, self.min_cohort = control, min_cohort
        self.cluster, self.n_clusters = cluster, n_clusters
        self.n_boot, self.random_state = n_boot, random_state

    # ------------------------------------------------------------------
    def exposure_times(self, panel: SpatialPanel, adopt: np.ndarray) -> np.ndarray:
        """First period in which a treated unit lies within ``exposure_distance`` (∞ if never)."""
        e = np.full(panel.n_units, np.inf)
        for t in panel.times:
            src = np.flatnonzero(adopt <= t)
            if len(src) == 0:
                continue
            tree = cKDTree(panel.unit_coords[src])
            hits = tree.query_ball_point(panel.unit_coords, r=self.exposure_distance)
            for i, h in enumerate(hits):
                if np.isinf(e[i]) and any(src[j] != i for j in h):
                    e[i] = t
        return e

    def _cells(self, Y, times, adopt, expo):
        """List of (kind, cohort, time, cohort_mask, base_index, time_index)."""
        tpos = {t: k for k, t in enumerate(times)}
        cells = []
        for kind, key in (("direct", adopt), ("spillover", expo)):
            if kind == "direct":
                cohorts = np.unique(adopt[np.isfinite(adopt)])
            else:
                cohorts = np.unique(expo[np.isfinite(expo) & np.isinf(adopt)])
            for g in cohorts:
                if kind == "direct":
                    mask = adopt == g
                else:
                    mask = (expo == g) & np.isinf(adopt)
                if mask.sum() < self.min_cohort or tpos[g] == 0:
                    continue
                b = tpos[g] - 1
                for k, t in enumerate(times):
                    if k == b:
                        continue
                    ref = max(t, times[b])
                    clean = (adopt > ref) & (expo > ref) & ~mask
                    if self.control == "never":
                        clean &= np.isinf(adopt)
                    if clean.sum() < self.min_cohort:
                        continue
                    cells.append((kind, g, t, mask, clean, b, k))
        return cells

    @staticmethod
    def _att(Y, cell, wts):
        _, _, _, mask, clean, b, k = cell
        d = Y[:, k] - Y[:, b]
        wt, wc = wts * mask, wts * clean
        if wt.sum() == 0 or wc.sum() == 0:
            return np.nan, 0.0
        return float(np.sum(wt * d) / wt.sum() - np.sum(wc * d) / wc.sum()), float(wt.sum())

    def _aggregate(self, Y, times, cells, wts):
        """Cell ATTs plus overall and event-time aggregates (weights = cohort size)."""
        rows = []
        for c in cells:
            att, size = self._att(Y, c, wts)
            rows.append((c[0], c[1], c[2], c[2] - c[1], att, size))
        df = pd.DataFrame(rows, columns=["kind", "cohort", "time", "rel_time", "att", "size"])
        agg = {}
        for kind in ("direct", "spillover"):
            d = df[(df.kind == kind) & df.att.notna()]
            post = d[d.rel_time >= 0]
            agg[(kind, "overall")] = np.average(post.att, weights=post["size"]) if len(post) and post["size"].sum() else np.nan
            for k, grp in d.groupby("rel_time"):
                agg[(kind, k)] = np.average(grp.att, weights=grp["size"]) if grp["size"].sum() else np.nan
        return df, agg

    # ------------------------------------------------------------------
    def fit(self, panel: SpatialPanel):
        if not panel.is_balanced:
            raise ValueError("StaggeredSpatialDiD needs a balanced panel")
        if panel.post is None:
            raise ValueError("panel needs post / treat_time to define adoption")
        Y, times = panel.wide(), panel.times
        adopt = panel.adoption_times()
        adopt = np.where(np.isnan(adopt), np.inf, adopt)
        expo = self.exposure_times(panel, adopt)
        self.adoption_, self.exposure_ = adopt, expo
        cells = self._cells(Y, times, adopt, expo)
        if not any(c[0] == "direct" for c in cells):
            raise ValueError("no estimable direct cohort (need a pre-period and clean controls)")
        n = panel.n_units
        ones = np.ones(n)
        self.cells_, agg = self._aggregate(Y, times, cells, ones)

        # bootstrap over clusters
        if self.cluster == "unit":
            cl = np.arange(n)
        else:
            G = max(50, n // 8) if self.n_clusters == "auto" else int(self.n_clusters)
            cl = SpatialBlockKFold(n_splits=2, n_blocks=min(G, n), random_state=self.random_state).make_blocks(
                panel.unit_coords)
        _, cl = np.unique(cl, return_inverse=True)
        n_cl = cl.max() + 1
        rng = np.random.default_rng(self.random_state)
        keys = list(agg)
        boot = np.full((self.n_boot, len(keys)), np.nan)
        for bi in range(self.n_boot):
            counts = np.bincount(rng.integers(0, n_cl, n_cl), minlength=n_cl).astype(float)
            _, a = self._aggregate(Y, times, cells, counts[cl])
            boot[bi] = [a.get(k, np.nan) for k in keys]
        se = dict(zip(keys, np.nanstd(boot, axis=0, ddof=1)))
        self._boot, self._keys = boot, keys

        z = stats.norm.ppf(0.975)
        es = [{"kind": k[0], "rel_time": k[1], "att": v, "se": se[k], "ci_lower": v - z * se[k],
               "ci_upper": v + z * se[k]} for k, v in agg.items() if k[1] != "overall"]
        self.event_study_ = pd.DataFrame(es).sort_values(["kind", "rel_time"]).reset_index(drop=True)

        effects = {"att": EffectEstimate("att", agg[("direct", "overall")], se[("direct", "overall")])}
        if np.isfinite(agg.get(("spillover", "overall"), np.nan)):
            effects["spillover"] = EffectEstimate("spillover", agg[("spillover", "overall")],
                                                  se[("spillover", "overall")])
        self.pretrend_test_ = self._pretrend(agg, keys, boot)

        from geolattice.spatial_analysis.spatialcausal.quasi import _cluster_vcov, _ols_twfe
        D = (panel.treated[panel.unit_index] * panel.post).astype(float)[:, None]
        b0, p0 = _ols_twfe(panel.y, D, panel.unit_index, panel.time_index)
        self.naive_twfe_ = EffectEstimate("att_naive_twfe", b0[0],
                                          np.sqrt(_cluster_vcov(p0, panel.unit_index)[0, 0]))
        effects["att_naive_twfe"] = self.naive_twfe_
        n_coh = self.cells_.groupby("kind")["cohort"].nunique().to_dict()
        self.result_ = SpatialCausalResult(
            "Staggered spatial DiD (group-time ATT)", effects, se_method=f"cluster bootstrap ({self.cluster})",
            n=panel.n_obs, exposure_distance=self.exposure_distance, control=self.control,
            cohorts=n_coh, never_treated=int(np.isinf(adopt).sum()),
            exposed_never_treated=int((np.isinf(adopt) & np.isfinite(expo)).sum()),
            pretrend_p_value=None if self.pretrend_test_ is None else round(self.pretrend_test_["p_value"], 4))
        return self

    def _pretrend(self, agg, keys, boot):
        idx = [j for j, k in enumerate(keys) if k[0] == "direct" and k[1] != "overall" and k[1] < -1]
        if not idx:
            return None
        b = np.array([agg[keys[j]] for j in idx])
        B = boot[:, idx]
        ok = np.all(np.isfinite(B), axis=1)
        if ok.sum() < len(idx) + 10:
            return None
        V = np.cov(B[ok], rowvar=False).reshape(len(idx), len(idx))
        W = float(b @ np.linalg.pinv(V) @ b)
        return {"wald": W, "df": len(idx), "p_value": float(stats.chi2.sf(W, len(idx)))}

    def effect(self, name: str = "att") -> float:
        return self.result_[name].estimate

    def summary(self) -> str:
        return self.result_.summary()
