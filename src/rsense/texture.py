"""Texture analysis for optical and radar (SAR) imagery.

Texture quantifies the spatial arrangement of pixel values, not just their magnitude, so
it separates classes that spectral/backscatter intensity alone cannot (urban fabric vs.
bare soil, crop types, forest structure, SAR speckle patterns from smooth water vs. rough
canopy). Every function here accepts a 2-D ``xr.DataArray`` (or plain NumPy array), or one
with leading dims (band/polarization/time), in which case the algorithm is applied
independently to each leading-dim slice. Pass one optical band or one SAR polarization at a
time for the clearest signal; SAR inputs are expected in linear or dB backscatter (same
convention as :mod:`geolattice.rsense.sar`).

Six families of methods, each covering both optical and radar imagery (see each function's
docstring for which surface/backscatter properties it responds to):

1. **Statistical** -- :func:`first_order_stats` (histogram moments + entropy, no spatial
   relationship), :func:`glcm_texture` (Haralick 1973 gray-level co-occurrence features),
   :func:`glrlm_texture` (gray-level run-length emphasis, directional/linear texture).
2. **Structural (syntactic)** -- :func:`edge_density` (density of structural
   primitives/edges), :func:`lacunarity` (gliding-box gap/clump distribution).
3. **Transform-based (spectral)** -- :func:`gabor_texture` (oriented/frequency filter
   bank), :func:`fourier_texture` (radial power-spectrum rings), :func:`wavelet_texture`
   (multi-resolution Haar subband energy).
4. **Local pattern-based** -- :func:`lbp_texture` (Local Binary Pattern histogram
   uniformity/entropy).
5. **Model-based** -- :func:`mrf_texture_energy` (Markov Random Field clique-energy
   proxy), :func:`fractal_dimension` (differential box-counting), :func:`autoregressive_texture`
   (local autoregressive fit).
6. **Deep-learning-based** -- :func:`deep_texture_features` (pretrained CNN backbone
   feature maps, needs ``pip install geolattice[rsense-dl]``).

Most functions accept a ``size`` (moving-window side, pixels) and, for the costlier
per-window algorithms (GLCM, GLRLM, Fourier, fractal dimension, LBP summary), a ``step``
that skips pixels for speed; the result is nearest-neighbour resampled back to the input's
full resolution either way, so every function returns a georeferenced raster the same shape
as its input. NaN input pixels stay NaN in the output for every per-pixel function
(:func:`first_order_stats`, :func:`edge_density`, :func:`lacunarity`, :func:`gabor_texture`,
:func:`wavelet_texture`, :func:`mrf_texture_energy`, :func:`autoregressive_texture`). The
``step``-based grid functions (:func:`glcm_texture`, :func:`glrlm_texture`,
:func:`fourier_texture`, :func:`fractal_dimension`, :func:`lbp_texture`) only skip a window
*centered* on a missing pixel -- the window's edge margin and the ``step``-pixel grid are
both then nearest-neighbour resampled back to full resolution, so an isolated single-pixel
NaN is approximated from nearby valid windows rather than reproduced exactly at that pixel.
"""

from __future__ import annotations

from typing import Any, Literal, Sequence

import numpy as np
import xarray as xr
from scipy.ndimage import uniform_filter, zoom as _ndzoom

from .backend import to_cpu
from .core._utils import local_mean_var, nan_box_mean

__all__ = [
    "first_order_stats",
    "glcm_texture",
    "glrlm_texture",
    "edge_density",
    "lacunarity",
    "gabor_texture",
    "fourier_texture",
    "wavelet_texture",
    "lbp_texture",
    "mrf_texture_energy",
    "fractal_dimension",
    "autoregressive_texture",
    "deep_texture_features",
]


# --------------------------------------------------------------------------- shared helpers

def _leading_loop(arr: np.ndarray, fn2d):
    """Apply ``fn2d`` (2-D array -> 2-D array, or 2-D array -> dict[str, 2-D array]) to the
    trailing two axes of ``arr``, looping over any leading dims (band/time/polarization)."""
    flat = arr.reshape((-1,) + arr.shape[-2:]) if arr.ndim > 2 else arr[None]
    first = fn2d(flat[0])
    if isinstance(first, dict):
        outs = {k: np.empty(flat.shape[:1] + np.shape(v), "float64") for k, v in first.items()}
        for k, v in first.items():
            outs[k][0] = v
        for i in range(1, flat.shape[0]):
            r = fn2d(flat[i])
            for k in outs:
                outs[k][i] = r[k]
        shp = arr.shape[:-2]
        return {k: (v.reshape(shp + v.shape[-2:]) if arr.ndim > 2 else v[0]) for k, v in outs.items()}
    out = np.empty(flat.shape[:1] + np.shape(first), "float64")
    out[0] = first
    for i in range(1, flat.shape[0]):
        out[i] = fn2d(flat[i])
    return out.reshape(arr.shape) if arr.ndim > 2 else out[0]


def _wrap(img: Any, result: Any, was_da: bool):
    """Rewrap a plain-array or dict-of-arrays result with ``img``'s coords/CRS if ``img``
    was a DataArray (each result keeps ``img``'s exact shape, so ``img.copy(data=...)``
    is enough -- no resampling or coordinate rebuilding needed)."""
    if isinstance(result, dict):
        if was_da:
            return xr.Dataset({k: img.copy(data=v.astype("float32")).rename(k) for k, v in result.items()})
        return {k: v.astype("float32") for k, v in result.items()}
    return img.copy(data=result.astype("float32")) if was_da else result.astype("float32")


def _as_cpu_array(img: Any) -> tuple[np.ndarray, bool]:
    was_da = isinstance(img, xr.DataArray)
    arr = np.asarray(to_cpu(img.data if was_da else img), dtype="float64")
    return arr, was_da


def _fill(a2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (finite mask, NaN-filled-with-local-mean copy) for an algorithm that cannot
    itself handle missing data."""
    finite = np.isfinite(a2d)
    fill_value = np.nanmean(a2d[finite]) if finite.any() else 0.0
    return finite, np.where(finite, a2d, fill_value)


def _quantize(a2d: np.ndarray, levels: int, lo: float | None = None, hi: float | None = None) -> np.ndarray:
    """Rescale to ``levels`` integer gray levels (global min/max unless given)."""
    finite = np.isfinite(a2d)
    lo = np.nanmin(a2d[finite]) if lo is None and finite.any() else (lo if lo is not None else 0.0)
    hi = np.nanmax(a2d[finite]) if hi is None and finite.any() else (hi if hi is not None else 1.0)
    if hi <= lo:
        hi = lo + 1.0
    q = np.clip((a2d - lo) / (hi - lo) * (levels - 1), 0, levels - 1)
    q = np.where(finite, q, 0)
    return q.astype("uint8" if levels <= 256 else "uint16")


def _zoom_to(small: np.ndarray, shape: tuple[int, int], order: int = 0) -> np.ndarray:
    """Resample a coarse ``step``-pixel grid back to full resolution."""
    if small.shape == shape:
        return small
    zy, zx = shape[0] / small.shape[0], shape[1] / small.shape[1]
    out = _ndzoom(small, (zy, zx), order=order, mode="nearest")
    ty, tx = shape
    oy, ox = out.shape
    if oy < ty or ox < tx:
        out = np.pad(out, ((0, max(0, ty - oy)), (0, max(0, tx - ox))), mode="edge")
    return out[:ty, :tx]


def _grid(h: int, w: int, pad: int, step: int) -> tuple[list[int], list[int]]:
    """Window-center rows/cols, ``step`` pixels apart, staying ``pad`` away from the edges."""
    gy = list(range(pad, max(h - pad, pad + 1), step)) or [min(pad, h - 1)]
    gx = list(range(pad, max(w - pad, pad + 1), step)) or [min(pad, w - 1)]
    return gy, gx


# =================================================================== 1. Statistical methods

def first_order_stats(img: Any, *, size: int = 9,
                       stats: Sequence[str] = ("mean", "std", "skewness", "kurtosis", "entropy"),
                       entropy_bins: int = 32) -> xr.Dataset | dict:
    """First-order (histogram-based) texture: per-pixel moments and entropy in a moving
    window, with no spatial/co-occurrence relationship between pixels -- the cheapest and
    least specific texture family, often used as a quick pre-filter before GLCM/LBP.

    ``mean``/``std``/``variance`` describe local brightness and spread; ``skewness`` and
    ``kurtosis`` describe the shape of the local value distribution (e.g. a mix of bright
    scatterers over a dark background skews a SAR window positively); ``entropy`` is the
    Shannon entropy (bits) of a local ``entropy_bins``-level histogram, high for
    disordered/heterogeneous texture (urban fabric, strong speckle) and low for uniform
    surfaces (water, bare soil, smooth agricultural fields). Works the same way on optical
    reflectance bands or SAR linear/dB backscatter.
    """
    arr, was_da = _as_cpu_array(img)
    finite = np.isfinite(arr)
    out: dict[str, np.ndarray] = {}
    needs_moments = {"mean", "variance", "std", "skewness", "kurtosis"} & set(stats)
    m = v = None
    if needs_moments:
        m, v = local_mean_var(arr, size, np)
    if "mean" in stats:
        out["mean"] = m
    if "variance" in stats:
        out["variance"] = v
    if "std" in stats:
        out["std"] = np.sqrt(v)
    if "skewness" in stats:
        sd = np.sqrt(np.maximum(v, 1e-12))
        out["skewness"] = nan_box_mean((arr - m) ** 3, size, np) / sd**3
    if "kurtosis" in stats:
        out["kurtosis"] = nan_box_mean((arr - m) ** 4, size, np) / np.maximum(v, 1e-12) ** 2 - 3.0
    if "entropy" in stats:
        from skimage.filters.rank import entropy as _rank_entropy
        from skimage.morphology import disk

        def ent(a2d):
            finite, filled = _fill(a2d)
            lo = float(np.nanmin(a2d[finite])) if finite.any() else 0.0
            hi = float(np.nanmax(a2d[finite])) if finite.any() else 1.0
            q = _quantize(filled, entropy_bins, lo, hi)
            e = _rank_entropy(q, disk(max(1, size // 2))).astype("float64")
            return np.where(finite, e, np.nan)

        out["entropy"] = _leading_loop(arr, ent)
    out = {k: np.where(finite, v, np.nan) for k, v in out.items()}
    return _wrap(img, out, was_da)


def glcm_texture(img: Any, *, size: int = 9, step: int = 2, distance: int = 1,
                  angles: Sequence[float] = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
                  levels: int = 32,
                  props: Sequence[str] = ("contrast", "homogeneity", "energy", "correlation",
                                           "dissimilarity", "entropy")) -> xr.Dataset | dict:
    """Second-order statistical texture via the Gray-Level Co-occurrence Matrix (Haralick
    et al. 1973) -- the most widely used texture method for satellite imagery, especially
    SAR, because it summarizes the spatial dependency between neighbouring gray levels
    rather than relying on absolute intensity.

    For each ``size``x``size`` window (every ``step`` pixels for speed, nearest-neighbour
    resampled back to full resolution), quantizes to ``levels`` gray levels and builds a
    symmetric, normalized GLCM averaged over ``angles`` at ``distance`` pixels. ``contrast``
    and ``dissimilarity`` are high for rough/heterogeneous texture (urban fabric, disturbed
    forest, rough SAR backscatter); ``homogeneity`` and ``energy`` are high for smooth,
    uniform texture (water, pasture, bare soil); ``correlation`` measures linear
    gray-tone dependency; ``entropy`` measures GLCM disorder. 32-64 gray levels is usually
    enough; more levels need a bigger window to populate the co-occurrence matrix reliably.
    """
    from skimage.feature import graycomatrix, graycoprops

    arr, was_da = _as_cpu_array(img)
    pad = size // 2
    angles = list(angles)

    def one(a2d):
        h, w = a2d.shape
        finite = np.isfinite(a2d)
        q = _quantize(a2d, levels)
        gy, gx = _grid(h, w, pad, step)
        grid = {p: np.full((len(gy), len(gx)), np.nan) for p in props}
        for i, y in enumerate(gy):
            for j, x in enumerate(gx):
                if not finite[y, x]:
                    continue
                win = q[max(0, y - pad):y + pad + 1, max(0, x - pad):x + pad + 1]
                if win.shape[0] < 2 or win.shape[1] < 2:
                    continue
                glcm = graycomatrix(win, distances=[distance], angles=angles, levels=levels,
                                     symmetric=True, normed=True)
                glcm = glcm.mean(axis=(2, 3), keepdims=True)  # average over distance/angle
                for p in props:
                    if p == "entropy":
                        pmat = glcm[..., 0, 0]
                        with np.errstate(divide="ignore", invalid="ignore"):
                            grid[p][i, j] = -np.sum(np.where(pmat > 0, pmat * np.log2(pmat), 0))
                    else:
                        grid[p][i, j] = graycoprops(glcm, p)[0, 0]
        return {p: _zoom_to(grid[p], (h, w), order=0) for p in props}

    result = _leading_loop(arr, one)
    return _wrap(img, result, was_da)


def glrlm_texture(img: Any, *, size: int = 16, step: int = 8, levels: int = 16,
                   directions: Sequence[str] = ("horizontal", "vertical"),
                   features: Sequence[str] = ("SRE", "LRE", "RP")) -> xr.Dataset | dict:
    """Gray-Level Run-Length Matrix texture: short/long-run emphasis and run percentage,
    sensitive to directional/linear patterns (furrowed fields, building rows, linear SAR
    layover/shadow streaks) that GLCM's pairwise statistics do not directly target.

    ``SRE`` (Short Run Emphasis) is high where runs of equal gray level are short (fine,
    busy texture); ``LRE`` (Long Run Emphasis) is high where runs are long (coarse,
    striped texture); ``RP`` (Run Percentage) is the fraction of pixels that start a new
    run (near 1 for fine/noisy texture, low for smooth or strongly directional texture).
    Computed every ``step`` pixels over a ``size``x``size`` window (quantized to
    ``levels`` gray levels first), then nearest-neighbour resampled back to full resolution.
    """
    arr, was_da = _as_cpu_array(img)
    pad = size // 2

    def run_lengths(line: np.ndarray) -> np.ndarray:
        if line.size == 0:
            return np.array([], dtype="float64")
        change = np.flatnonzero(np.diff(line) != 0)
        starts = np.concatenate(([0], change + 1))
        ends = np.concatenate((change + 1, [line.size]))
        return (ends - starts).astype("float64")

    def one(a2d):
        h, w = a2d.shape
        finite = np.isfinite(a2d)
        q = _quantize(a2d, levels)
        gy, gx = _grid(h, w, pad, step)
        grid = {f: np.full((len(gy), len(gx)), np.nan) for f in features}
        n_dirs = len(directions)
        for i, y in enumerate(gy):
            for j, x in enumerate(gx):
                if not finite[y, x]:
                    continue
                win = q[max(0, y - pad):y + pad + 1, max(0, x - pad):x + pad + 1]
                runs: list[float] = []
                if "horizontal" in directions:
                    for row in win:
                        runs.extend(run_lengths(row))
                if "vertical" in directions:
                    for col in win.T:
                        runs.extend(run_lengths(col))
                if not runs:
                    continue
                r = np.asarray(runs, dtype="float64")
                n_pix = win.size * n_dirs
                if "SRE" in features:
                    grid["SRE"][i, j] = np.mean(1.0 / r**2)
                if "LRE" in features:
                    grid["LRE"][i, j] = np.mean(r**2)
                if "RP" in features:
                    grid["RP"][i, j] = r.size / n_pix
        return {f: _zoom_to(grid[f], (h, w), order=0) for f in features}

    result = _leading_loop(arr, one)
    return _wrap(img, result, was_da)


# =========================================================== 2. Structural (syntactic) methods

def edge_density(img: Any, *, size: int = 9, sigma: float = 1.0,
                  low_threshold: float | None = None,
                  high_threshold: float | None = None) -> xr.DataArray | np.ndarray:
    """Structural/syntactic texture: local density of Canny edges (Canny 1986) -- the
    fraction of edge pixels inside a moving ``size``x``size`` window. Structural texture
    methods treat an image as primitives (edges, lines) arranged by placement rules, so
    density-of-primitives is the simplest such descriptor.

    High values mark regularly repeating structural elements and their boundaries
    (building rows, field furrows, road networks, forest-edge patchwork, double-bounce
    edges in urban SAR); low values mark texture-less or smoothly continuous surfaces
    (open water, bare soil, pasture). ``sigma``/``low_threshold``/``high_threshold`` tune
    the underlying Canny detector (``None`` thresholds let scikit-image pick automatically
    from the image's gradient statistics).
    """
    from skimage.feature import canny

    arr, was_da = _as_cpu_array(img)

    def one(a2d):
        finite, filled = _fill(a2d)
        edges = canny(filled, sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold)
        density = nan_box_mean(np.where(finite, edges.astype("float64"), np.nan), size, np)
        return np.where(finite, density, np.nan)

    result = _leading_loop(arr, one)
    return _wrap(img, result, was_da)


def lacunarity(img: Any, *, box_sizes: Sequence[int] = (2, 4, 8, 16),
               size: int | None = None) -> xr.Dataset | dict:
    """Structural texture via gliding-box lacunarity (Plotnick et al. 1996):
    ``lambda(r) = Var(box mass) / Mean(box mass)^2 + 1`` for box side ``r``, computed over
    gliding boxes within a moving analysis window of ``size`` pixels (default
    ``4 * max(box_sizes)``). Unlike GLCM/first-order stats, lacunarity is about *how
    gaps and clumps are arranged* at scale ``r``, independent of the pixel values' own
    distribution -- useful for structural patterns like patchy canopy gaps, scattered
    buildings, or sparse strong scatterers in SAR.

    High lacunarity at scale ``r`` means mass is unevenly distributed at that scale
    (clumped/gappy); lacunarity near 1 means a spatially uniform pattern at that scale.
    Returns one variable per box size (``lac_r2``, ``lac_r4``, ...).
    """
    arr, was_da = _as_cpu_array(img)
    win = size or max(box_sizes) * 4

    def one(a2d):
        finite, filled = _fill(a2d)
        out = {}
        for r in box_sizes:
            mass = uniform_filter(filled, size=r, mode="reflect") * (r * r)
            m, v = local_mean_var(mass, win, np)
            lac = v / np.maximum(m, 1e-9) ** 2 + 1.0
            out[f"lac_r{r}"] = np.where(finite, lac, np.nan)
        return out

    result = _leading_loop(arr, one)
    return _wrap(img, result, was_da)


# ========================================================= 3. Transform-based (spectral) methods

def gabor_texture(img: Any, *, frequencies: Sequence[float] = (0.1, 0.25, 0.4),
                   thetas: Sequence[float] = (0, 45, 90, 135), size: int = 9) -> xr.Dataset | dict:
    """Transform-based texture via a bank of Gabor filters (Gabor 1946; Jain & Farrokhnia
    1991): each (frequency, orientation) filter responds to texture repeating at that
    spatial frequency and direction, and the local mean energy of its response in a moving
    ``size``x``size`` window is the texture feature for that filter.

    Crop rows, orchard grids, and oriented SAR speckle/layover patterns show strong,
    orientation-specific responses; isotropic textures (undifferentiated forest, uniform
    SAR clutter) respond similarly at every orientation. ``thetas`` are in degrees.
    """
    from skimage.filters import gabor

    arr, was_da = _as_cpu_array(img)
    thetas = list(thetas)
    frequencies = list(frequencies)

    def one(a2d):
        finite, filled = _fill(a2d)
        lo, hi = np.percentile(filled, [2, 98])
        norm = np.clip((filled - lo) / max(hi - lo, 1e-9), 0, 1)
        out = {}
        for f in frequencies:
            for t in thetas:
                real, imag = gabor(norm, frequency=f, theta=np.radians(t))
                energy = real**2 + imag**2
                name = f"gabor_f{f:g}_t{int(t)}".replace(".", "p")
                pooled = nan_box_mean(np.where(finite, energy, np.nan), size, np)
                out[name] = np.where(finite, pooled, np.nan)
        return out

    result = _leading_loop(arr, one)
    return _wrap(img, result, was_da)


def fourier_texture(img: Any, *, size: int = 32, step: int = 16, n_rings: int = 6) -> xr.Dataset | dict:
    """Transform-based texture via the 2-D Fourier power spectrum (Rosenfeld & Troy 1970):
    for each ``size``x``size`` tile (every ``step`` pixels), computes a Hann-windowed 2-D
    FFT power spectrum and bins it into ``n_rings`` radial frequency rings.

    Periodic textures (row crops, orchard grids, regularly spaced buildings, periodic
    ocean-wave backscatter patterns) concentrate energy in specific rings; speckle-
    dominated or unstructured texture spreads energy evenly across rings. Returns the
    fraction of total spectral energy in each ring (``ring0`` = lowest spatial frequency,
    i.e. coarsest pattern), nearest-neighbour resampled back to full resolution from the
    ``step``-pixel grid.
    """
    arr, was_da = _as_cpu_array(img)
    pad = size // 2
    window2d = np.outer(np.hanning(size), np.hanning(size))
    yy, xx = np.mgrid[0:size, 0:size] - size // 2
    radius = np.sqrt(yy**2 + xx**2)
    edges = np.linspace(0, radius.max(), n_rings + 1)
    edges[-1] += 1e-9  # include the exact corner pixel(s) at radius.max() in the last ring
    ring_names = [f"ring{k}" for k in range(n_rings)]

    def one(a2d):
        h, w = a2d.shape
        finite = np.isfinite(a2d)
        fill_value = np.nanmean(a2d[finite]) if finite.any() else 0.0
        gy, gx = _grid(h, w, pad, step)
        grid = {name: np.full((len(gy), len(gx)), np.nan) for name in ring_names}
        for i, y in enumerate(gy):
            for j, x in enumerate(gx):
                if not finite[y, x]:
                    continue
                tile = a2d[y - pad:y - pad + size, x - pad:x - pad + size]
                if tile.shape != (size, size):
                    continue
                tile = np.where(np.isfinite(tile), tile, fill_value)
                spec = np.abs(np.fft.fftshift(np.fft.fft2(tile * window2d))) ** 2
                total = spec.sum() or 1.0
                for k, name in enumerate(ring_names):
                    m = (radius >= edges[k]) & (radius < edges[k + 1])
                    grid[name][i, j] = spec[m].sum() / total
        return {name: _zoom_to(grid[name], (h, w), order=0) for name in ring_names}

    result = _leading_loop(arr, one)
    return _wrap(img, result, was_da)


def _haar_step(a: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One level of an orthonormal 2-D Haar DWT: approximation LL and detail subbands
    LH (vertical detail), HL (horizontal detail), HH (diagonal detail), each half size."""
    h, w = a.shape
    a = a[: h - h % 2, : w - w % 2]
    even_rows, odd_rows = a[0::2, :], a[1::2, :]
    low = (even_rows + odd_rows) / np.sqrt(2)
    high = (even_rows - odd_rows) / np.sqrt(2)
    low_even, low_odd = low[:, 0::2], low[:, 1::2]
    high_even, high_odd = high[:, 0::2], high[:, 1::2]
    ll = (low_even + low_odd) / np.sqrt(2)
    lh = (low_even - low_odd) / np.sqrt(2)
    hl = (high_even + high_odd) / np.sqrt(2)
    hh = (high_even - high_odd) / np.sqrt(2)
    return ll, lh, hl, hh


def wavelet_texture(img: Any, *, level: int = 2, size: int = 9) -> xr.Dataset | dict:
    """Transform-based, multi-resolution texture via a 2-D Haar discrete wavelet transform
    (Mallat 1989), implemented directly in NumPy (no extra dependency needed beyond
    ``geolattice[rsense]``). Each decomposition level splits the image into an approximation
    (``LL``, carried into the next level) and three detail subbands at half the previous
    resolution: ``LH`` (horizontal edges/texture), ``HL`` (vertical), ``HH`` (diagonal).

    Returns the local mean energy of each detail subband at each level (upsampled back to
    full resolution) as a scale- and orientation-specific texture feature -- useful because
    real textures (forest canopy, urban fabric, SAR speckle) carry structure at several
    scales at once, unlike a single-scale GLCM/Gabor window.
    """
    arr, was_da = _as_cpu_array(img)

    def one(a2d):
        finite, a = _fill(a2d)
        h, w = a.shape
        out = {}
        cur = a
        for lvl in range(1, level + 1):
            ll, lh, hl, hh = _haar_step(cur)
            for name, sub in (("LH", lh), ("HL", hl), ("HH", hh)):
                energy = _zoom_to(sub**2, (h, w), order=1)
                pooled = nan_box_mean(np.where(finite, energy, np.nan), size, np)
                out[f"{name}_L{lvl}"] = np.where(finite, pooled, np.nan)
            cur = ll
        return out

    result = _leading_loop(arr, one)
    return _wrap(img, result, was_da)


# ================================================================ 4. Local pattern-based methods

def lbp_texture(img: Any, *, radius: int = 1, n_points: int = 8,
                 method: Literal["uniform", "default", "ror", "var"] = "uniform",
                 size: int = 9, step: int = 2, levels: int = 256,
                 features: Sequence[str] = ("uniformity", "entropy")) -> xr.Dataset | dict:
    """Local Binary Pattern texture (Ojala, Pietikainen & Maenpaa 2002): each pixel is
    compared to ``n_points`` neighbours sampled on a circle of ``radius`` pixels, giving a
    micro-pattern code per pixel (rotation-invariant with ``method="uniform"``). The local
    histogram of codes in a moving ``size``x``size`` window (every ``step`` pixels) is
    summarized as ``uniformity`` (histogram energy -- high for repetitive micro-texture)
    and/or ``entropy`` (high for disordered micro-texture).

    Cheap relative to GLCM and effective for building-density/impervious-surface mapping
    and for separating speckle-dominated SAR texture classes without GLCM's directional
    windows. The input is rescaled to ``levels`` gray levels first (LBP compares
    neighbouring values, so this just avoids floating-point noise creating spurious
    micro-patterns; raise ``levels`` for very smooth, low-contrast scenes). Result is
    nearest-neighbour resampled back to full resolution.
    """
    from skimage.feature import local_binary_pattern

    arr, was_da = _as_cpu_array(img)
    n_bins = n_points + 2 if method == "uniform" else 2**n_points
    pad = size // 2

    def one(a2d):
        finite, filled = _fill(a2d)
        codes = local_binary_pattern(_quantize(filled, levels), n_points, radius, method=method)
        h, w = codes.shape
        padded = np.pad(codes, pad, mode="reflect")
        gy = [min(y, h - 1) for y in range(0, h, step)] or [0]
        gx = [min(x, w - 1) for x in range(0, w, step)] or [0]
        names = [f"lbp_{f}" for f in features]
        grid = {name: np.full((len(gy), len(gx)), np.nan) for name in names}
        for i, y in enumerate(gy):
            for j, x in enumerate(gx):
                if not finite[y, x]:
                    continue
                win = padded[y:y + size, x:x + size].ravel()
                hist = np.bincount(win.astype("int64"), minlength=n_bins).astype("float64")
                p = hist / hist.sum()
                if "uniformity" in features:
                    grid["lbp_uniformity"][i, j] = np.sum(p**2)
                if "entropy" in features:
                    with np.errstate(divide="ignore", invalid="ignore"):
                        grid["lbp_entropy"][i, j] = -np.sum(np.where(p > 0, p * np.log2(p), 0))
        return {name: _zoom_to(grid[name], (h, w), order=0) for name in names}

    result = _leading_loop(arr, one)
    return _wrap(img, result, was_da)


# ======================================================================= 5. Model-based methods

def mrf_texture_energy(img: Any, *, size: int = 9, neighborhood: Literal[4, 8] = 4
                        ) -> xr.DataArray | np.ndarray:
    """Model-based texture via a Markov Random Field clique-energy proxy. Many
    unsupervised SAR/optical segmentation methods model a pixel's value as conditionally
    dependent on its nearest neighbours under a Gibbs/Ising smoothness prior; the local
    mean squared disagreement with those neighbours (pooled over a ``size``x``size``
    window) is a simple, well-behaved estimate of how strongly that prior is violated --
    i.e. how "textured" the neighbourhood is in the sense the MRF model cares about.

    Low energy means the neighbourhood is well explained by the smoothness prior (smooth,
    homogeneous surfaces: water, pasture, calm backscatter); high energy means strong local
    disagreement (edges, rough/heterogeneous surfaces, strong SAR speckle).
    """
    arr, was_da = _as_cpu_array(img)
    shifts4 = ((0, 1), (0, -1), (1, 0), (-1, 0))
    shifts8 = shifts4 + ((1, 1), (1, -1), (-1, 1), (-1, -1))
    shifts = shifts8 if neighborhood == 8 else shifts4

    def one(a2d):
        finite, a = _fill(a2d)
        sq_diff = np.zeros_like(a)
        for dy, dx in shifts:
            shifted = np.roll(np.roll(a, dy, axis=0), dx, axis=1)
            sq_diff += (a - shifted) ** 2
        sq_diff /= len(shifts)
        pooled = nan_box_mean(np.where(finite, sq_diff, np.nan), size, np)
        return np.where(finite, pooled, np.nan)

    result = _leading_loop(arr, one)
    return _wrap(img, result, was_da)


def fractal_dimension(img: Any, *, size: int = 32, step: int = 16,
                       box_sizes: Sequence[int] = (2, 4, 8, 16)) -> xr.DataArray | np.ndarray:
    """Model-based texture via the local fractal dimension, estimated per tile with
    differential box-counting (Sarkar & Chaudhuri 1994): treats the intensity surface as a
    3-D relief (x, y, intensity) and counts how many boxes of side ``r`` are needed to
    cover it, for each ``r`` in ``box_sizes``, within a ``size``x``size`` tile (every
    ``step`` pixels; ``box_sizes`` must all divide ``size`` evenly). The fractal dimension
    is the slope of log(box count) vs. log(1/r).

    Rougher, more irregular surfaces (natural terrain, disturbed/heterogeneous canopy,
    rough SAR backscatter) have a higher fractal dimension (closer to 3) than smooth,
    man-made, or spatially uniform surfaces (closer to 2).
    """
    arr, was_da = _as_cpu_array(img)
    pad = size // 2
    valid_r = [r for r in box_sizes if size % r == 0]

    def dbc(tile: np.ndarray) -> float:
        t = tile - tile.min()
        t = t / max(t.max(), 1e-9) * size  # rescale intensity to the same range as the spatial grid
        counts = []
        for r in valid_r:
            n = size // r
            boxed = t[: n * r, : n * r].reshape(n, r, n, r)
            lo = boxed.min(axis=(1, 3))
            hi = boxed.max(axis=(1, 3))
            counts.append(float(np.sum(np.ceil(hi / r) - np.floor(lo / r) + 1)))
        if len(counts) < 2:
            return np.nan
        x = np.log(1.0 / np.asarray(valid_r, "float64"))
        y = np.log(np.asarray(counts, "float64"))
        return float(np.polyfit(x, y, 1)[0])

    def one(a2d):
        h, w = a2d.shape
        finite = np.isfinite(a2d)
        gy, gx = _grid(h, w, pad, step)
        grid = np.full((len(gy), len(gx)), np.nan)
        for i, y in enumerate(gy):
            for j, x in enumerate(gx):
                if not finite[y, x]:
                    continue
                tile = a2d[y - pad:y - pad + size, x - pad:x - pad + size]
                if tile.shape != (size, size) or not np.isfinite(tile).all():
                    continue
                grid[i, j] = dbc(tile)
        return _zoom_to(grid, (h, w), order=0)

    result = _leading_loop(arr, one)
    return _wrap(img, result, was_da)


def autoregressive_texture(img: Any, *, size: int = 9) -> xr.Dataset | dict:
    """Model-based texture via a simplified isotropic autoregressive fit: regresses each
    pixel on the mean of its 4 nearest neighbours, ``x ~= rho * mean_4nn(x) + e``, with
    ``rho`` and the residual found in closed form from local sums within a moving
    ``size``x``size`` window (the classical AR/SAR texture-model idea, simplified to one
    isotropic coefficient instead of a full multi-directional neighbourhood fit).

    ``rho`` close to 1 means strong, smooth spatial correlation (coarse, slowly varying
    texture); ``rho`` close to 0 means the pixel looks independent of its neighbours
    (fine texture or noise/speckle). ``residual`` is the AR(1) fit's unexplained variance,
    normalized by the local variance -- high where this simple model fits poorly.
    """
    arr, was_da = _as_cpu_array(img)

    def one(a2d):
        finite, a = _fill(a2d)
        nbr = (np.roll(a, 1, 0) + np.roll(a, -1, 0) + np.roll(a, 1, 1) + np.roll(a, -1, 1)) / 4.0
        mx, vx = local_mean_var(a, size, np)
        mn, vn = local_mean_var(nbr, size, np)
        # Cov(X, Y) = E[XY] - E[X]E[Y] (the box-filter identity local_mean_var also uses for
        # variance) -- NOT the mean of (x - mean(x)) * (y - mean(y)), which, for each pixel,
        # mixes that pixel's own-window mean into every *other* pixel's term in the box sum
        # and collapses to ~0 whenever x is locally smooth (e.g. a ramp), not just at x's center.
        cov = nan_box_mean(a * nbr, size, np) - mx * mn
        rho = np.clip(cov / np.maximum(vn, 1e-9), -1.5, 1.5)
        resid = nan_box_mean((a - mx - rho * (nbr - mn)) ** 2, size, np) / np.maximum(vx, 1e-9)
        return {"rho": np.where(finite, rho, np.nan), "residual": np.where(finite, resid, np.nan)}

    result = _leading_loop(arr, one)
    return _wrap(img, result, was_da)


# ================================================================ 6. Deep-learning-based methods

def deep_texture_features(img: Any, *, model_name: str = "resnet18", layer: int = 1,
                           reduce: Literal["map", "mean"] = "map",
                           device: str | None = None, pretrained: bool = True
                           ) -> xr.Dataset | dict:
    """Deep-learning texture features from a pretrained CNN backbone (via ``timm``): runs
    ``img`` through the backbone's stem up to stage ``layer`` (0-indexed) and returns that
    stage's feature-map channels as texture features, upsampled (bilinear) to ``img``'s
    resolution. Early CNN layers behave like a learned, data-driven filter bank
    (conceptually similar to the Gabor filters above, but learned from natural images
    rather than hand-designed), and can outperform hand-crafted texture features for
    complex, non-stationary optical or SAR textures, given enough downstream training data.

    ``img`` needs 3 input channels, since ImageNet-pretrained backbones expect RGB: a
    single optical/SAR band is replicated across all 3 channels, 2 bands get a repeated
    first band as the third channel, and more than 3 bands are truncated to the first 3 --
    pass your 3 most informative bands/polarizations explicitly for anything but a quick
    look. Each channel is independently percentile-stretched to [0, 1] before the forward
    pass. With ``reduce="mean"``, returns one scalar per feature channel (a compact texture
    descriptor for the whole array) instead of a per-pixel map.

    Needs ``pip install geolattice[rsense-dl]`` (PyTorch + timm). Runs a single forward pass
    over the whole input, not tiled -- clip large scenes to the AOI first, or tile them
    with :func:`geolattice.rsense.core.apply_tiled` and call this per tile.
    """
    from .dl._torch import require_torch, torch_device

    torch = require_torch()
    try:
        import timm
    except ImportError as e:
        raise ImportError("Deep texture features need timm: `pip install geolattice[rsense-dl]`") from e

    was_da = isinstance(img, xr.DataArray)
    if isinstance(img, xr.Dataset):
        img = img.to_array("band")
        was_da = True
    if was_da and "band" in img.dims:
        spatial_dims = tuple(d for d in img.dims if d != "band")
        arr = np.asarray(to_cpu(img.transpose("band", *spatial_dims)).values, dtype="float32")
    else:
        base = np.asarray(to_cpu(img.data if was_da else img), dtype="float32")
        arr = base[None] if base.ndim == 2 else base
        spatial_dims = img.dims[-2:] if was_da else None

    c, h, w = arr.shape
    finite = np.all(np.isfinite(arr), axis=0)
    arr = np.nan_to_num(arr)
    if c == 1:
        arr = np.repeat(arr, 3, axis=0)
    elif c == 2:
        arr = np.concatenate([arr, arr[:1]], axis=0)
    elif c > 3:
        arr = arr[:3]
    for ch in range(3):
        lo, hi = np.percentile(arr[ch], [2, 98])
        arr[ch] = np.clip((arr[ch] - lo) / max(hi - lo, 1e-9), 0, 1)

    model = timm.create_model(model_name, pretrained=pretrained, features_only=True, out_indices=(layer,))
    dev = torch_device(device)
    model.to(dev).eval()
    with torch.no_grad():
        x = torch.from_numpy(arr[None]).to(dev)
        feat = model(x)[0][0].to("cpu").numpy()  # (channels, h', w')

    co, ho, wo = feat.shape
    upsampled = np.stack([_ndzoom(feat[k], (h / ho, w / wo), order=1, mode="nearest") for k in range(co)])
    upsampled = upsampled[:, :h, :w]

    if reduce == "mean":
        return {f"deep_{k}_mean": float(np.nanmean(np.where(finite, upsampled[k], np.nan)))
                for k in range(co)}
    maps = {f"deep_{k}": np.where(finite, upsampled[k], np.nan).astype("float32") for k in range(co)}
    if not was_da:
        return maps
    coords = {d: img[d] for d in spatial_dims if d in img.coords}
    ds = xr.Dataset({k: xr.DataArray(v, dims=spatial_dims, coords=coords, name=k) for k, v in maps.items()})
    if img.rio.crs is not None:
        ds = ds.rio.write_crs(img.rio.crs)
    return ds
