"""Season-long UAV phenotyping: per-image traits across flight dates and growth curves.

:func:`trait_table` applies a trait function to every frame in a catalogue
(:func:`geolattice.rsense.uav.build_catalog`) and :func:`summarize_dates` averages per date.
:func:`fit_logistic` fits ``y = base + amp / (1 + exp(-rate (t - t_mid)))`` to a trait such
as canopy cover or height and returns the inflection date and maximum growth rate.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd


def trait_table(
    catalog: pd.DataFrame,
    func: Callable[[Any], dict[str, float]],
    *,
    path_col: str = "path",
    reader: Callable[[str], Any] | None = None,
    errors: str = "warn",
    progress: bool = False,
    **read_kw: Any,
) -> pd.DataFrame:
    """Run ``func(image) -> {trait: value}`` on each catalogue row and join the results.

    ``reader`` defaults to :func:`geolattice.rsense.uav.read_frame` (``read_kw`` passed on, e.g.
    ``max_dim=2048`` or ``gsd=0.00185``). ``errors``: ``"raise"``, ``"warn"`` or ``"ignore"``.
    """
    import warnings

    from .io import read_frame

    read = reader or (lambda p: read_frame(p, **read_kw))
    it = catalog.iterrows()
    if progress:
        from tqdm.auto import tqdm

        it = tqdm(it, total=len(catalog), desc="traits")
    rows = []
    for i, r in it:
        try:
            rows.append({"_row": i, **func(read(r[path_col]))})
        except Exception as e:  # noqa: BLE001
            if errors == "raise":
                raise
            if errors == "warn":
                warnings.warn(f"{r[path_col]}: {e}", stacklevel=2)
    traits = (
        pd.DataFrame(rows).set_index("_row")
        if rows
        else pd.DataFrame(index=pd.Index([], name="_row"))
    )
    return catalog.join(traits, how="inner" if errors != "raise" else "left")


def summarize_dates(
    table: pd.DataFrame, traits: list[str] | None = None, *, by: list[str] | None = None
) -> pd.DataFrame:
    """Mean, SD and count of each trait per flight date (and per ``by`` columns, e.g. crop)."""
    keys = (by or ["crop"]) + ["date"]
    keys = [k for k in keys if k in table]
    traits = traits or [c for c in table.select_dtypes("number").columns if c not in keys]
    g = table.groupby(keys, dropna=False)[traits].agg(["mean", "std", "count"])
    g.columns = [f"{t}_{s}" for t, s in g.columns]
    return g.reset_index()


def _days(dates: Any, origin: Any = None) -> tuple[np.ndarray, pd.Timestamp]:
    d = pd.to_datetime(pd.Series(dates))
    o = pd.Timestamp(origin) if origin is not None else d.min()
    return (d - o).dt.total_seconds().to_numpy() / 86400.0, o


def logistic(t: np.ndarray, base: float, amp: float, rate: float, t_mid: float) -> np.ndarray:
    return base + amp / (1 + np.exp(-rate * (t - t_mid)))


def fit_logistic(
    dates: Any,
    values: Any,
    *,
    origin: Any = None,
    decreasing: bool = False,
    min_transition_days: float | None = None,
    limits: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Fit a 4-parameter logistic curve to a trait over time.

    Returns ``base, amp, rate, t_mid`` (days since ``origin``), ``mid_date``,
    ``max_rate`` (trait units per day, at the inflection), ``r2`` and a ``predict(dates)``
    function. ``decreasing=True`` starts the fit for a declining trait (e.g. greenness
    during senescence). Needs at least 4 distinct dates.

    Noisy traits sampled weekly can otherwise collapse into a step between two flights. The
    rate is therefore bounded so the 10-90 % transition (``4.39 / rate`` days) lasts at least
    ``min_transition_days`` (default: twice the median gap between flight dates), and the
    mid-point is kept within one data span of the observed dates. ``limits=(lo, hi)`` keeps
    both asymptotes inside a physical range, e.g. ``(0, 1)`` for a fraction, which matters
    when the data cover only the start of a change.
    """
    from scipy.optimize import curve_fit

    t, o = _days(dates, origin)
    y = np.asarray(values, float)
    ok = np.isfinite(t) & np.isfinite(y)
    t, y = t[ok], y[ok]
    uniq = np.unique(t)
    if uniq.size < 4:
        raise ValueError("need values on at least 4 distinct dates")
    lo, hi = float(np.min(y)), float(np.max(y))
    span = (t.max() - t.min()) or 1.0
    if min_transition_days is None:
        min_transition_days = 2 * float(np.median(np.diff(uniq)))
    rate_max = 4.39 / max(min_transition_days, 1e-6)  # 10-90 % lasts 2 ln 9 / rate days
    # fit in terms of the two asymptotes (start, end) so both can be bounded directly
    def curve(tt: np.ndarray, y0: float, y1: float, r: float, m: float) -> np.ndarray:
        return logistic(tt, y0, y1 - y0, r, m)

    yr = (hi - lo) or 1.0
    a_lo, a_hi = (lo - 2 * yr, hi + 2 * yr) if limits is None else map(float, limits)
    start, end = (hi, lo) if decreasing else (lo, hi)
    p0 = [
        float(np.clip(start, a_lo, a_hi)),
        float(np.clip(end, a_lo, a_hi)),
        min(8 / span, rate_max / 2),
        float(np.median(t)),
    ]
    bounds = ([a_lo, a_lo, 1e-6, t.min() - span], [a_hi, a_hi, rate_max, t.max() + span])
    q, _ = curve_fit(curve, t, y, p0=p0, bounds=bounds, maxfev=20000)
    p = np.array([q[0], q[1] - q[0], q[2], q[3]])
    base, amp, rate, tm = map(float, p)
    pred = logistic(t, *p)
    ss = np.sum((y - y.mean()) ** 2)
    r2 = 1 - np.sum((y - pred) ** 2) / ss if ss > 0 else np.nan

    def predict(d: Any) -> np.ndarray:
        return logistic(_days(d, o)[0], *p)

    return {
        "base": base,
        "amp": amp,
        "rate": rate,
        "t_mid": tm,
        "mid_date": o + pd.Timedelta(days=tm),
        "max_rate": amp * rate / 4,
        "r2": float(r2),
        "origin": o,
        "predict": predict,
    }


def growth_rates(
    table: pd.DataFrame, trait: str, *, date: str = "date", by: list[str] | None = None
) -> pd.DataFrame:
    """Change in ``trait`` per day between consecutive flight dates (per ``by`` group)."""
    keys = [k for k in (by or []) if k in table]
    t = table.sort_values(keys + [date]).copy()
    grp = t.groupby(keys) if keys else [(None, t)]
    parts = []
    for _, g in grp:
        g = g.copy()
        dd = g[date].diff().dt.total_seconds() / 86400.0
        g[f"{trait}_per_day"] = g[trait].diff() / dd
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def detect_stages(
    fit: dict[str, Any], *, levels: dict[str, float] | None = None
) -> pd.DataFrame:
    """Dates at which a fitted logistic curve reaches given fractions of its amplitude.

    ``fit`` is the output of :func:`fit_logistic`. ``levels`` maps a stage name to a
    fraction of the way from ``base`` to ``base + amp`` (defaults: ``onset`` 0.1, ``mid``
    0.5, ``plateau`` 0.9); for a declining trait the same fractions measure the decline.
    Returns one row per stage with ``day`` (since the fit origin), ``date`` and ``value``.
    """
    levels = levels or {"onset": 0.1, "mid": 0.5, "plateau": 0.9}
    rate, tm = fit["rate"], fit["t_mid"]
    rows = []
    for name, f in levels.items():
        if not 0 < f < 1:
            raise ValueError(f"level {name}={f} must lie strictly between 0 and 1")
        # base + amp * f  <=>  1 / (1 + exp(-rate (t - tm))) = f
        t = tm - np.log(1 / f - 1) / rate
        rows.append(
            {
                "stage": name,
                "fraction": f,
                "day": float(t),
                "date": fit["origin"] + pd.Timedelta(days=float(t)),
                "value": fit["base"] + fit["amp"] * f,
            }
        )
    return pd.DataFrame(rows).sort_values("day", ignore_index=True)


def frame_traits(
    image: Any,
    *,
    indices: tuple[str, ...] = ("ExG", "VARI", "TGI"),
    mask_index: str = "ExG",
    threshold: float | None = None,
    health: bool = True,
    count: bool = False,
    min_distance: float = 8.0,
    sigma: float = 2.0,
    hue_max: float = 75.0,
) -> dict[str, float]:
    """Standard RGB(+NIR) traits of one frame, ready for :func:`trait_table`.

    Canopy cover from an Otsu (or fixed ``threshold``) cut of ``mask_index``; the mean of
    each of ``indices`` (plus NDVI when a ``nir`` band exists) inside the vegetation mask;
    with ``health`` the senescent fraction of plant pixels (colour rule with ``hue_max``,
    see :func:`geolattice.rsense.uav.classify_senescence`); with ``count`` the number of plants
    found as ``mask_index`` peaks (``min_distance`` and ``sigma`` in the image's coordinate
    units: pixels for plain frames).

    Late in the season Otsu on ExG can split green from yellow leaves rather than plants
    from soil; pass a fixed ``threshold`` (e.g. 0.05 to 0.1 for ExG on 0-1 RGB) there.
    """
    from .canopy import canopy_cover, count_plants
    from .indices import vegetation_indices

    names = list(dict.fromkeys([mask_index, *indices]))
    if "band" in image.dims and "nir" in list(image.band.values):
        names.append("NDVI")
    vi = vegetation_indices(image, names)
    mask, cover = canopy_cover(vi[mask_index], threshold)
    out: dict[str, float] = {"canopy_cover": cover}
    for n in names:
        v = vi[n].where(mask)
        out[f"{n}_veg_mean"] = float(v.mean(skipna=True))
    if health:
        from .health import classify_senescence

        lab, _ = classify_senescence(image, np.asarray(mask.values, bool), hue_max=hue_max)
        plant = int((lab > 0).sum())
        out["senescent_frac"] = float((lab == 2).sum() / plant) if plant else np.nan
    if count:
        pts = count_plants(vi[mask_index], min_distance_m=min_distance, sigma_m=sigma)
        out["plant_count"] = float(len(pts))
    return out
