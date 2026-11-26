"""Correcting UAV digital elevation models made without ground control.

Without GCPs or RTK, Structure-from-Motion DEMs bend into a dome or bowl and tilt. Over a
flat field that systematic shape is far larger than the crop canopy. :func:`ransac_flatten`
fits a low-order polynomial surface to a reference DEM with RANSAC (so plants and other
outliers do not pull the fit) and removes the same surface from every date, keeping
heights comparable across the season (after Chaudhry, Rana & Jaleel 2026).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from os import PathLike
from typing import Any, Literal

import numpy as np
import xarray as xr


@dataclass
class FlattenResult:
    """Output of :func:`ransac_flatten`."""

    dems: list[xr.DataArray]
    trend: xr.DataArray
    valid: xr.DataArray
    ground: float | None
    info: dict[str, Any] = field(default_factory=dict)


def _as_da(dem: Any) -> xr.DataArray:
    if isinstance(dem, (str, PathLike)):
        from ..core.raster import read

        dem = read(dem)
    if isinstance(dem, xr.DataArray):
        return dem.squeeze("band", drop=True) if "band" in dem.dims else dem
    from .io import grid_coords

    a = np.asarray(dem, dtype=np.float32)
    return xr.DataArray(a, dims=("y", "x"), coords=grid_coords(*a.shape))


def trend_surface(
    dem: Any,
    *,
    degree: int = 2,
    valid: Any = None,
    max_points: int = 200_000,
    residual_threshold: float | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, dict]:
    """Fit ``z = poly(col, row)`` of ``degree`` to a DEM with RANSAC; return surface and fit info.

    ``max_points`` subsamples large DEMs for the fit (the surface is evaluated everywhere).
    Needs scikit-learn.
    """
    try:
        from sklearn.linear_model import LinearRegression, RANSACRegressor
        from sklearn.preprocessing import PolynomialFeatures
    except ImportError as e:
        raise ImportError("ransac_flatten needs `pip install scikit-learn`") from e
    z = np.asarray(getattr(dem, "values", dem), np.float64)
    ok = np.isfinite(z) if valid is None else (np.asarray(valid, bool) & np.isfinite(z))
    rows, cols = np.nonzero(ok)
    if rows.size < 10 * (degree + 1) ** 2:
        raise ValueError("too few valid pixels to fit the trend surface")
    rng = np.random.default_rng(seed)
    if rows.size > max_points:
        pick = rng.choice(rows.size, max_points, replace=False)
        rows, cols = rows[pick], cols[pick]
    scale = float(max(z.shape))  # keep polynomial terms well conditioned
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    X = poly.fit_transform(np.column_stack([cols, rows]) / scale)
    model = RANSACRegressor(
        LinearRegression(), residual_threshold=residual_threshold, random_state=seed
    ).fit(X, z[rows, cols])
    rr, cc = np.mgrid[0 : z.shape[0], 0 : z.shape[1]]
    surf = model.predict(poly.transform(np.column_stack([cc.ravel(), rr.ravel()]) / scale))
    surf = surf.reshape(z.shape).astype(np.float32)
    info = {
        "degree": degree,
        "inlier_frac": float(model.inlier_mask_.mean()),
        "trend_range": float(np.nanmax(surf[ok]) - np.nanmin(surf[ok])),
    }
    return surf, info


def ransac_flatten(
    dems: Sequence[Any],
    *,
    degree: int = 2,
    reference: int = 0,
    edge_threshold: float | None = -200.0,
    strength: float = 1.0,
    normalize: Literal["ground", "none", "lums"] = "ground",
    ground_percentile: float = 0.5,
    crop_to_common: bool = True,
    match: Literal["mean", "ground"] = "mean",
) -> FlattenResult:
    """Remove a shared doming/tilt surface from one or more DEMs of the same field.

    Parameters
    ----------
    dems : DEMs as GeoTIFF paths, DataArrays or 2-D arrays, ideally on the same grid.
    degree : polynomial order of the trend (2 captures doming; 1 only tilt).
    reference : index of the DEM the trend is fitted on (then applied to all of them).
    edge_threshold : pixels at or below this value are invalid (photogrammetry nodata
        fill such as -32767). ``None`` keeps all finite pixels.
    strength : fraction of the trend to remove (1 = all).
    normalize : ``ground`` subtracts the season-wide ``ground_percentile`` so bare soil sits
        near 0 m and heights stay in metres; ``none`` only matches each DEM's mean to the
        reference mean; ``lums`` reproduces the original script's 0-5 rescaling (unitless).
    crop_to_common : crop DEMs of different shapes to the smallest common shape (top-left),
        as the original script does. Prefer DEMs exported on one grid.
    match : how dates are levelled to the reference after detrending. ``mean`` (the LUMS
        script) matches each DEM's mean, which assumes the mean surface does not change, so
        a growing canopy is pulled down and growth is underestimated. ``ground`` matches each
        DEM's ``ground_percentile`` instead (bare soil must be visible on every date).

    Returns a :class:`FlattenResult` with the corrected DEMs, the trend surface, the
    combined valid mask, the ground level used and fit diagnostics.
    """
    das = [_as_da(d) for d in dems]
    shapes = {d.shape for d in das}
    if len(shapes) > 1:
        if not crop_to_common:
            raise ValueError(f"DEMs differ in shape {sorted(shapes)}; resample them to one grid")
        r = min(s[0] for s in shapes)
        c = min(s[1] for s in shapes)
        das = [d.isel(y=slice(0, r), x=slice(0, c)) for d in das]
    arrs = [np.asarray(d.values, np.float32) for d in das]
    valid = np.logical_and.reduce(
        [
            np.isfinite(a) & ((a > edge_threshold) if edge_threshold is not None else True)
            for a in arrs
        ]
    )
    if valid.sum() < 100:
        raise ValueError("fewer than 100 pixels valid in every DEM; check edge_threshold")
    trend, info = trend_surface(arrs[reference], degree=degree, valid=valid)
    if match not in ("mean", "ground"):
        raise ValueError("match must be 'mean' or 'ground'")

    def level(x: np.ndarray) -> float:
        v = x[valid]
        return float(v.mean()) if match == "mean" else float(np.percentile(v, ground_percentile))

    ref_level = level(arrs[reference] - strength * trend)
    corrected = []
    for a in arrs:
        c = a - strength * trend
        c = c + (ref_level - level(c))
        corrected.append(np.where(valid, c, np.nan).astype(np.float32))
    ground = None
    if normalize != "none":
        allv = np.concatenate([c[valid] for c in corrected])
        ground = float(np.percentile(allv, ground_percentile))
        if normalize == "lums":
            top = float(allv.max())
            s = 5.0 / (top - ground) if top > ground else 1.0
            corrected = [(c - ground) * s for c in corrected]
            info["lums_scale"] = s
        else:
            corrected = [c - ground for c in corrected]
    out = [
        d.copy(data=c)
        .rename("dem_flat")
        .assign_attrs(
            units="m" if normalize != "lums" else "relative", flattened=f"ransac degree {degree}"
        )
        for d, c in zip(das, corrected)
    ]
    return FlattenResult(
        out,
        das[reference].copy(data=trend).rename("trend"),
        das[reference].copy(data=valid).rename("valid"),
        ground,
        {**info, "reference": reference, "normalize": normalize, "match": match},
    )


def doming_profile(dem: Any, *, axis: Literal["x", "y"] = "x", n: int = 100) -> xr.DataArray:
    """Median elevation along one axis: a quick look at doming before and after flattening."""
    da = _as_da(dem)
    other = "y" if axis == "x" else "x"
    prof = da.median(other, skipna=True)
    if prof.sizes[axis] > n:
        prof = prof.coarsen({axis: prof.sizes[axis] // n}, boundary="trim").mean()
    return prof.rename(f"median_{da.name or 'z'}")


def write_dems(result: FlattenResult, paths: Sequence[str | PathLike]) -> list[str]:
    """Write corrected DEMs to GeoTIFF (georeferenced inputs keep their CRS)."""
    out = []
    for da, p in zip(result.dems, paths):
        da.astype("float32").rio.to_raster(p)
        out.append(str(p))
    return out


def synthetic_dem(
    shape: tuple[int, int] = (300, 450),
    *,
    gsd: float = 0.05,
    canopy_height: float = 0.6,
    doming: float = 1.5,
    tilt: tuple[float, float] = (0.4, -0.25),
    base: float = 10.0,
    plots: tuple[int, int] = (3, 5),
    plot_gap: float = 0.5,
    height_spread: float = 0.3,
    noise: float = 0.02,
    seed: int = 0,
) -> tuple[xr.DataArray, xr.DataArray]:
    """A GCP-free style DSM of a plot trial: crop canopy on a domed, tilted ground.

    The field is split into ``plots`` (rows, cols) with alleys of bare soil (``plot_gap``
    is the fraction of each plot cell left bare; RANSAC needs bare ground to be the
    majority to lock onto it). Each plot gets its own canopy height around ``canopy_height`` (spread ``height_spread`` as a fraction),
    with a rough, plant-scale texture. ``doming`` is the bulge (m) between centre and corner
    and ``tilt`` the (x, y) rise in m across the field. Use it to test :func:`ransac_flatten`
    when no real DEM is at hand. Returns ``(dsm, true_chm)`` in metres on a ``gsd`` grid.
    """
    from scipy.ndimage import gaussian_filter

    from .io import grid_coords

    rng = np.random.default_rng(seed)
    rows, cols = shape
    yy, xx = np.mgrid[0:rows, 0:cols]
    u = 2 * xx / max(cols - 1, 1) - 1  # -1 .. 1 across the field
    v = 2 * yy / max(rows - 1, 1) - 1
    trend = base + doming * (1 - (u**2 + v**2) / 2) + tilt[0] * u / 2 + tilt[1] * v / 2

    pr, pc = plots
    chm = np.zeros(shape, np.float32)
    gap_r, gap_c = int(plot_gap * rows / pr), int(plot_gap * cols / pc)
    heights = canopy_height * (1 + height_spread * (rng.random(pr * pc) - 0.5) * 2)
    texture = gaussian_filter(rng.random(shape), max(1.0, 0.04 / gsd))
    texture = (texture - texture.min()) / (np.ptp(texture) or 1.0)
    for i in range(pr):
        for j in range(pc):
            r0, r1 = i * rows // pr + gap_r // 2, (i + 1) * rows // pr - gap_r // 2
            c0, c1 = j * cols // pc + gap_c // 2, (j + 1) * cols // pc - gap_c // 2
            h = heights[i * pc + j]
            chm[r0:r1, c0:c1] = h * (0.75 + 0.5 * texture[r0:r1, c0:c1])
    dsm = trend + chm + rng.normal(0, noise, shape)
    coords = grid_coords(rows, cols, gsd)
    attrs = {"units": "m", "gsd_m": gsd, "plots": f"{pr}x{pc}"}
    return (
        xr.DataArray(dsm.astype(np.float32), dims=("y", "x"), coords=coords, name="dsm", attrs=attrs),
        xr.DataArray(chm, dims=("y", "x"), coords=coords, name="chm_true", attrs=attrs),
    )
