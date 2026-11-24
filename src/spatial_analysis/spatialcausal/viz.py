"""Plotting helpers (requires matplotlib: ``pip install geolattice[spatialcausal-viz]``)."""
from __future__ import annotations

from typing import Optional

import numpy as np

from geolattice.spatial_analysis.spatialcausal._utils import as_coords


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("plotting needs matplotlib: pip install geolattice[spatialcausal-viz]") from exc
    return plt


def plot_effect_map(coords, values, ax=None, title: str = "Estimated CATE", cmap: str = "RdBu_r",
                    center: Optional[float] = None, s: float = 12, **kwargs):
    """Scatter map of unit-level effects (e.g. ``SpatialDRLearner.cate_``)."""
    plt = _plt()
    coords = as_coords(coords)
    values = np.asarray(values, float)
    ax = ax or plt.subplots(figsize=(6, 5))[1]
    if center is not None:
        from matplotlib.colors import TwoSlopeNorm
        kwargs.setdefault("norm", TwoSlopeNorm(vcenter=center))
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=values, cmap=cmap, s=s, **kwargs)
    plt.colorbar(sc, ax=ax, shrink=0.8)
    ax.set_title(title)
    ax.set_aspect("equal")
    return ax


def plot_exposure_conditions(coords, conditions, ax=None, s: float = 12):
    """Map of the four exposure conditions from an ``ExposureMapping``."""
    plt = _plt()
    from geolattice.spatial_analysis.spatialcausal.exposure import CONDITIONS, CONDITION_LABELS
    coords = as_coords(coords)
    conditions = np.asarray(conditions)
    ax = ax or plt.subplots(figsize=(6, 5))[1]
    colors = ["#bdbdbd", "#6baed6", "#fd8d3c", "#a50f15"]
    for c, col in enumerate(colors):
        m = conditions == c
        ax.scatter(coords[m, 0], coords[m, 1], s=s, c=col, label=CONDITION_LABELS[CONDITIONS[c]])
    ax.legend(fontsize=8, loc="best")
    ax.set_aspect("equal")
    ax.set_title("Exposure conditions")
    return ax


def plot_effects(result, truth: Optional[dict] = None, ax=None):
    """Forest plot of contrasts in a ``SpatialCausalResult`` (true values as ×)."""
    plt = _plt()
    df = result.to_frame()
    ax = ax or plt.subplots(figsize=(6, 0.6 * len(df) + 1.2))[1]
    yy = np.arange(len(df))[::-1]
    ax.errorbar(df["estimate"], yy, xerr=[df["estimate"] - df["ci_lower"], df["ci_upper"] - df["estimate"]],
                fmt="o", capsize=4, color="#2b5c8a")
    if truth:
        tv = [truth.get(k, np.nan) for k in df.index]
        ax.scatter(tv, yy, marker="x", color="#c0392b", zorder=3, label="truth")
        ax.legend()
    ax.axvline(0, color="grey", lw=0.8, ls="--")
    ax.set_yticks(yy)
    ax.set_yticklabels(df.index)
    ax.set_title(result.name)
    return ax


def plot_cutoff_sensitivity(estimator_factory, data, cutoffs, ax=None, effect: str = "ate"):
    """Refit with ``se='conley'`` over several cutoffs and plot the CI width."""
    plt = _plt()
    est, lo, hi = [], [], []
    for c in cutoffs:
        r = estimator_factory(c).fit(data).result_[effect]
        est.append(r.estimate)
        lo.append(r.ci[0])
        hi.append(r.ci[1])
    ax = ax or plt.subplots(figsize=(6, 4))[1]
    ax.fill_between(cutoffs, lo, hi, alpha=0.3)
    ax.plot(cutoffs, est, "o-")
    ax.set_xlabel("Conley cutoff")
    ax.set_ylabel(effect)
    ax.set_title("Sensitivity of the CI to the Conley cutoff")
    return ax


# ----------------------------------------------------------------------------
# v0.2 plots
# ----------------------------------------------------------------------------
def plot_local_effects(coords, local_effects, ax=None, column: str = "theta", alpha_ns: float = 0.25,
                       title: str = "Local treatment effect", cmap: str = "RdBu_r", s: float = 14, **kwargs):
    """Map of local effects (``GWDML.local_effects_`` or ``GWCausalForest.local_ate_``).

    Units whose 95% CI covers zero are drawn faded.
    """
    plt = _plt()
    coords = as_coords(coords)
    df = local_effects
    col = column if column in df else ("ate" if "ate" in df else df.columns[0])
    sig = (df["ci_lower"] > 0) | (df["ci_upper"] < 0)
    ax = ax or plt.subplots(figsize=(6, 5))[1]
    vmin, vmax = kwargs.pop("vmin", df[col].min()), kwargs.pop("vmax", df[col].max())
    ax.scatter(coords[~sig, 0], coords[~sig, 1], c=df[col][~sig], cmap=cmap, vmin=vmin, vmax=vmax,
               s=s, alpha=alpha_ns, **kwargs)
    sc = ax.scatter(coords[sig, 0], coords[sig, 1], c=df[col][sig], cmap=cmap, vmin=vmin, vmax=vmax, s=s, **kwargs)
    plt.colorbar(sc, ax=ax, shrink=0.8)
    ax.set_title(f"{title}\n(faded: 95% CI includes 0)")
    ax.set_aspect("equal")
    return ax


def plot_rdd(rdd, y=None, n_bins: int = 30, ax=None, title: str = "Geographic RDD"):
    """Binned outcome means against signed distance to the border, with local linear fits."""
    plt = _plt()
    t, yy, _, _, _ = rdd._data
    yy = yy if y is None else np.asarray(y, float)
    r, h = rdd.running_, rdd.bandwidth_
    ax = ax or plt.subplots(figsize=(6, 4))[1]
    lim = np.quantile(np.abs(r), 0.95)
    edges = np.linspace(-lim, lim, n_bins + 1)
    mids = 0.5 * (edges[1:] + edges[:-1])
    idx = np.digitize(r, edges) - 1
    means = np.array([yy[idx == b].mean() if np.any(idx == b) else np.nan for b in range(n_bins)])
    ax.scatter(mids, means, color="#555", s=18, zorder=3)
    for side, col in ((0, "#2b5c8a"), (1, "#c0392b")):
        m = (t == side) & (np.abs(r) <= h)
        if m.sum() > 2:
            b = np.polyfit(r[m], yy[m], 1)
            xx = np.linspace(0, h, 50) * (1 if side else -1)
            ax.plot(xx, np.polyval(b, xx), color=col, lw=2)
    ax.axvline(0, color="k", lw=0.8)
    ax.axvspan(-h, h, color="grey", alpha=0.08)
    ax.set_xlabel("signed distance to border (treated side > 0)")
    ax.set_ylabel("outcome")
    ax.set_title(f"{title}: jump = {rdd.effect():.3f}")
    return ax


def plot_event_study(did, ax=None, title: str = "Event study: treated and spillover rings"):
    """Group × relative-time coefficients from ``SpatialDiD(event_study=True)``."""
    plt = _plt()
    df = did.event_study_
    ax = ax or plt.subplots(figsize=(7, 4))[1]
    groups = list(dict.fromkeys(df["group"]))
    offs = np.linspace(-0.25, 0.25, len(groups))
    for o, g in zip(offs, groups):
        d = df[df.group == g]
        ax.errorbar(d.rel_time + o, d.estimate, yerr=1.96 * d.se, fmt="o-", ms=4, capsize=2, label=g)
    ax.axhline(0, color="grey", lw=0.8)
    ax.axvline(-0.5, color="k", ls="--", lw=0.8)
    ax.set_xlabel("periods relative to treatment (reference = -1)")
    ax.set_ylabel("effect")
    ax.legend(fontsize=8)
    ax.set_title(title)
    return ax


def plot_synthetic_control(sc, ax=None):
    """Observed vs synthetic outcome (top) and the gap (bottom)."""
    plt = _plt()
    if ax is None:
        _, ax = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    ax[0].plot(sc.observed_.index, sc.observed_, color="k", lw=2, label="treated")
    ax[0].plot(sc.synthetic_.index, sc.synthetic_, color="#c0392b", ls="--", lw=2, label="synthetic")
    ax[0].axvline(sc.treat_time_, color="grey", ls=":")
    ax[0].legend()
    ax[0].set_title(f"Spatial synthetic control (buffer = {sc.buffer:g}): ATT = {sc.att_:.3f}")
    if hasattr(sc, "placebos_") and len(sc.placebos_):
        Y, pre = sc._Y, sc._pre
        for u in sc.placebos_.unit:
            j = int(np.flatnonzero(sc._panel.units == u)[0])
            donors, _ = sc._donor_pool(j, sc._other_treated | {sc._target}, sc.buffer)
            if len(donors) >= 2:
                ax[1].plot(sc.gap_.index, sc._fit_one(Y, pre, j, donors)[2], color="grey", alpha=0.25, lw=0.8)
    ax[1].plot(sc.gap_.index, sc.gap_, color="k", lw=2)
    ax[1].axhline(0, color="grey", lw=0.8)
    ax[1].axvline(sc.treat_time_, color="grey", ls=":")
    ax[1].set_ylabel("gap (grey: placebos)")
    return ax
