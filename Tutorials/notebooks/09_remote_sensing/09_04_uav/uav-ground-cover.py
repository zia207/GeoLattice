#!/usr/bin/env python3
"""
Python conversion of ``uav-ground-cover.Rmd``.

Green Ground Cover (GC) from UAV images, after Maas & Rajan (2008).

Workflow (mirrors the R Markdown):
    1. Load the 5-band UAV stack and plot a false-colour composite (R=NIR, G=Red, B=Green).
    2. Extract red (band 2) and NIR (band 5) digital numbers.
    3. Fit the Bare Soil Line with the "minimum" method of ``landsat::BSL()``.
    4. Compute the Perpendicular Vegetation Index (PVI) and PVI at full canopy (PVI.FC).
    5. Compute Ground Cover (GC = PVI / PVI.FC).
    6. Compute the Weighted Difference Vegetation Index (WDVI).
    7. Save maps (PNG) and index rasters (GeoTIFF).

The Bare Soil Line uses major axis (MA) regression exactly as implemented in
``lmodel2::lmodel2`` (the default method used by ``landsat::BSL``), so the
coefficients match the original R output.

Usage:
    python uav-ground-cover.py
    python uav-ground-cover.py multi_bwh_mc_20150227.tif -o outputs_20150227
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import linregress

# Band layout of multi_bwh_mc_*.tif (1-based, as in the R code)
BAND_RED = 2
BAND_NIR = 5
BAND_GREEN = 3

# R's BSL() default: values >= maxval are excluded from the soil-line fit
MAXVAL = 255
# R's BSL() default upper quantile used to locate the full-cover vegetation point
ULIMIT = 0.99

# R colour names used by scale_fill_gradientn(): red, yellow, green, green4
GC_COLORS = ["#FF0000", "#FFFF00", "#00FF00", "#008B00"]
GC_CMAP = LinearSegmentedColormap.from_list("gc", GC_COLORS)


# --------------------------------------------------------------------------- #
# I/O helpers
# --------------------------------------------------------------------------- #
def load_stack(path: str | Path) -> tuple[np.ndarray, dict, object]:
    """Read all bands as float64 with nodata converted to NaN."""
    with rasterio.open(path) as src:
        stack = src.read(masked=True).astype("float64")
        profile = src.profile.copy()
        transform = src.transform
    return stack.filled(np.nan), profile, transform


def valid_mask(*arrays: np.ndarray) -> np.ndarray:
    """True where every input array is finite."""
    mask = np.isfinite(arrays[0])
    for arr in arrays[1:]:
        mask &= np.isfinite(arr)
    return mask


def write_geotiff(path: str | Path, data: np.ndarray, profile: dict) -> None:
    """Write a single-band float32 GeoTIFF with NaN as nodata."""
    out_profile = {
        "driver": "GTiff",
        "height": profile["height"],
        "width": profile["width"],
        "count": 1,
        "dtype": "float32",
        "crs": profile.get("crs"),
        "transform": profile["transform"],
        "nodata": np.nan,
        "compress": "lzw",
    }
    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(data.astype("float32"), 1)


# --------------------------------------------------------------------------- #
# Bare Soil Line
# --------------------------------------------------------------------------- #
def bare_soil_line(
    red: np.ndarray,
    nir: np.ndarray,
    maxval: int = MAXVAL,
    ulimit: float = ULIMIT,
) -> dict:
    """
    Port of ``landsat::BSL(method = "minimum")``.

    Returns the MA regression coefficients and the full-cover vegetation point.
    """
    mask = valid_mask(red, nir)
    r = red[mask].ravel()
    n = nir[mask].ravel()

    # BSL(): bsl.joint[apply(bsl.joint, 1, function(x) all(x < maxval)), ]
    keep = (r < maxval) & (n < maxval)
    r, n = r[keep], n[keep]

    # factor(red, levels = seq(1, maxval)); only exact integer red levels match
    integer = np.isclose(r, np.rint(r))
    r_int = np.rint(r[integer]).astype(np.int32)
    n_int = n[integer]

    levels = np.unique(r_int[(r_int >= 1) & (r_int <= maxval - 1)])
    # split(NIR, red) -> min(NIR) for each red level
    x = levels.astype("float64")
    y = np.array([n_int[r_int == level].min() for level in levels], dtype="float64")

    # Ordinary least squares, used by lmodel2 to derive the MA slope
    ols = linregress(x, y)
    b_ols = float(ols.slope)
    r_val = float(ols.rvalue)
    rsquare = r_val**2

    # lmodel2::MA.reg (major axis / principal axis slope)
    d = (b_ols**2 - rsquare) / (rsquare * b_ols)
    slope = 0.5 * (d + np.sign(r_val) * np.sqrt(d**2 + 4.0))
    intercept = float(y.mean() - slope * x.mean())

    # Full-cover vegetation point: among the highest NIR/red ratios, take the
    # pixel minimising red + (maxval - NIR) (i.e. high NIR and low red).
    ratio = n / r
    threshold = np.quantile(ratio, ulimit)
    candidates = ratio > threshold
    score = r[candidates] + (maxval - n[candidates])
    best = int(np.argmin(score))
    top_red = float(r[candidates][best])
    top_nir = float(n[candidates][best])

    return {
        "slope": slope,
        "intercept": intercept,
        "r": r_val,
        "r_squared": rsquare,
        "top_red": top_red,
        "top_nir": top_nir,
        "n_levels": int(levels.size),
        "n_pixels": int(r.size),
    }


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def stretch_linear(band: np.ndarray, quantiles: tuple[float, float] = (0.02, 0.98)) -> np.ndarray:
    """RStoolbox::.stretch(method = 'lin'): 2-98 % linear stretch, clipped 0-1."""
    values = band[np.isfinite(band)]
    lo, hi = np.quantile(values, quantiles)
    if hi == lo:
        hi = lo + 1e-9
    out = (band - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def plot_false_color(stack: np.ndarray, transform, path: Path) -> None:
    """ggRGB(s, r = 5, g = 2, b = 3, stretch = 'lin')."""
    channels = [stretch_linear(stack[b - 1]) for b in (BAND_NIR, BAND_RED, BAND_GREEN)]
    rgb = np.dstack(channels)
    alpha = np.isfinite(stack[BAND_NIR - 1]).astype("float32")
    rgba = np.dstack([np.nan_to_num(rgb, nan=0.0), alpha])

    height, width = stack.shape[1], stack.shape[2]
    fig, ax = plt.subplots(figsize=(5, 5.5))
    ax.imshow(
        rgba,
        extent=(transform.c, transform.c + transform.a * width,
                transform.f + transform.e * height, transform.f),
        interpolation="nearest",
    )
    ax.set_axis_off()
    ax.set_title(
        "UAV's False Color Vegetation of Wheat Field\n"
        "(R= NIR, G= Red,  B=Green)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_bare_soil_line(
    red: np.ndarray,
    nir: np.ndarray,
    bsl: dict,
    path: Path,
) -> None:
    """Scatter of NIR vs red with the fitted soil line and vegetation peak."""
    mask = valid_mask(red, nir)
    x = red[mask].ravel()
    y = nir[mask].ravel()

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.scatter(x, y, s=1.0, c="grey", alpha=0.35, linewidths=0, rasterized=True)
    line_x = np.array([0.0, MAXVAL])
    ax.plot(
        line_x,
        bsl["slope"] * line_x + bsl["intercept"],
        color="red",
        linewidth=1.5,
    )
    ax.scatter(
        [bsl["top_red"]],
        [bsl["top_nir"]],
        c="green",
        s=80,
        zorder=5,
    )
    ax.text(150, 65, "a1", fontsize=9)
    ax.text(165, 65, "=", fontsize=9)
    ax.text(175, 65, f"{bsl['slope']:.3f}", fontsize=9)
    ax.text(150, 50, "a0", fontsize=9)
    ax.text(165, 50, "=", fontsize=9)
    ax.text(175, 50, f"{bsl['intercept']:.3f}", fontsize=9)

    ax.set_xlim(0, MAXVAL)
    ax.set_ylim(0, MAXVAL)
    ax.set_xlabel("RED-band")
    ax.set_ylabel("NIR-band")
    ax.set_title("Bare Soil Line")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_index(
    data: np.ndarray,
    transform,
    title: str,
    label: str,
    path: Path,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """ggR(...) + scale_fill_gradientn(colours = c('red','yellow','green','green4'))."""
    finite = data[np.isfinite(data)]
    vmin = float(finite.min()) if vmin is None else vmin
    vmax = float(finite.max()) if vmax is None else vmax

    extent = (
        transform.c,
        transform.c + transform.a * data.shape[1],
        transform.f + transform.e * data.shape[0],
        transform.f,
    )
    fig, ax = plt.subplots(figsize=(5, 5.5))
    im = ax.imshow(
        np.ma.masked_invalid(data),
        extent=extent,
        cmap=GC_CMAP,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.set_axis_off()
    ax.set_title(title, fontsize=11)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label(label)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        nargs="?",
        default="multi_bwh_mc_20150212.tif",
        help="5-band UAV GeoTIFF (default: %(default)s)",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        default="outputs",
        help="directory for rasters and figures (default: %(default)s)",
    )
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = base / input_path
    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = base / outdir
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- Load data -------------------------------------------------------- #
    stack, profile, transform = load_stack(input_path)
    red = stack[BAND_RED - 1]
    nir = stack[BAND_NIR - 1]

    # ---- Bare Soil Line --------------------------------------------------- #
    bsl = bare_soil_line(red, nir)
    a1 = bsl["slope"]
    a0 = bsl["intercept"]
    a2 = np.sqrt(1.0 + a1**2)

    print(f"Input image : {input_path.name}")
    print(f"Valid pixels: {bsl['n_pixels']:,} (red levels used: {bsl['n_levels']})")
    print("Bare Soil Line (minimum method, MA regression)")
    print(f"  Intercept (a0) = {a0:.6f}")
    print(f"  Slope     (a1) = {a1:.6f}")
    print(f"  R-squared      = {bsl['r_squared']:.4f}")
    print(f"  Full-cover point: Red = {bsl['top_red']:.0f}, NIR = {bsl['top_nir']:.0f}")

    # ---- Perpendicular Vegetation Index ----------------------------------- #
    pvi = (nir - a1 * red - a0) / a2
    pvi[pvi < 0.0] = 0.0
    pvi[pvi > 100.001] = 100.0
    pvi_norm = pvi / 100.0

    pvi_fc = (bsl["top_nir"] - a1 * bsl["top_red"] - a0) / a2
    print(f"  PVI at full cover (PVI.FC) = {pvi_fc:.4f}")

    # ---- Ground Cover ----------------------------------------------------- #
    gc = pvi / pvi_fc
    gc[gc > 1.00001] = 1.0
    gc[gc < 0.0] = 0.0

    # ---- Weighted Difference Vegetation Index ----------------------------- #
    wdvi = nir / MAXVAL - (a1 * red / MAXVAL)
    wdvi[wdvi > 1.0] = 1.0
    wdvi[wdvi < 0.0] = 0.0

    valid = valid_mask(red, nir)
    print(
        "Ground cover: "
        f"min = {np.nanmin(gc[valid]):.3f}, "
        f"mean = {np.nanmean(gc[valid]):.3f}, "
        f"max = {np.nanmax(gc[valid]):.3f}"
    )
    print(
        "WDVI        : "
        f"min = {np.nanmin(wdvi[valid]):.3f}, "
        f"max = {np.nanmax(wdvi[valid]):.3f}"
    )

    # ---- Figures ---------------------------------------------------------- #
    plot_false_color(stack, transform, outdir / "false_color_rgb.png")
    plot_bare_soil_line(red, nir, bsl, outdir / "bare_soil_line.png")
    plot_index(
        pvi_norm, transform,
        "Perpendicular Vegetation Index (PVI)", "PVI",
        outdir / "pvi.png",
    )
    plot_index(
        gc, transform,
        "Green Vegetation Cover (GC)", "GC",
        outdir / "ground_cover.png",
    )
    plot_index(
        wdvi, transform,
        "Weighted Vegetation Index (WDVI)", "WDVI",
        outdir / "wdvi.png",
    )

    # ---- Rasters ---------------------------------------------------------- #
    write_geotiff(outdir / "pvi.tif", pvi, profile)
    write_geotiff(outdir / "pvi_normalized.tif", pvi_norm, profile)
    write_geotiff(outdir / "ground_cover.tif", gc, profile)
    write_geotiff(outdir / "wdvi.tif", wdvi, profile)

    print(f"\nOutputs written to {outdir}")


if __name__ == "__main__":
    main()
