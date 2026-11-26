"""Spectral libraries: a small container plus readers for USGS splib07, ECOSTRESS/ASTER and
plain CSV, extraction from images, and resampling to a sensor's bands."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr


@dataclass
class SpectralLibrary:
    """Spectra (rows) on a shared wavelength grid (nm), with names and optional metadata."""

    wavelengths: np.ndarray
    spectra: np.ndarray
    names: list[str]
    metadata: pd.DataFrame = field(default_factory=pd.DataFrame)

    def __post_init__(self):
        self.wavelengths = np.asarray(self.wavelengths, float)
        self.spectra = np.atleast_2d(np.asarray(self.spectra, float))
        if self.spectra.shape != (len(self.names), len(self.wavelengths)):
            raise ValueError(f"spectra {self.spectra.shape} vs {len(self.names)} names × "
                             f"{len(self.wavelengths)} wavelengths")

    def __len__(self) -> int:
        return len(self.names)

    def __repr__(self) -> str:
        return (f"<SpectralLibrary {len(self)} spectra, {len(self.wavelengths)} bands "
                f"{self.wavelengths.min():.0f}–{self.wavelengths.max():.0f} nm>")

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.spectra, index=self.names, columns=np.round(self.wavelengths, 3))

    def to_dict(self) -> dict[str, np.ndarray]:
        return {n: s for n, s in zip(self.names, self.spectra)}

    def subset(self, names: Iterable[str] | str) -> SpectralLibrary:
        """Spectra whose names match (list of names, or a regex string)."""
        if isinstance(names, str):
            keep = [i for i, n in enumerate(self.names) if re.search(names, n, re.I)]
        else:
            want = set(names)
            keep = [i for i, n in enumerate(self.names) if n in want]
        md = self.metadata.iloc[keep] if len(self.metadata) else self.metadata
        return SpectralLibrary(self.wavelengths, self.spectra[keep],
                               [self.names[i] for i in keep], md)

    def resample(self, target: xr.DataArray | Sequence[float],
                 fwhm: Sequence[float] | None = None) -> SpectralLibrary:
        """Convolve to a cube's bands (Gaussian SRFs when FWHM is known) or to a wavelength list."""
        from .preprocess import _gaussian_matrix

        if isinstance(target, xr.DataArray):
            from .preprocess import resample_library

            wl = np.asarray(target["wavelength"].values, float)
            return SpectralLibrary(wl, resample_library(self.spectra, self.wavelengths, target),
                                   self.names, self.metadata)
        wl = np.asarray(target, float)
        if fwhm is None:
            S = np.stack([np.interp(wl, self.wavelengths, s, left=np.nan, right=np.nan)
                          for s in self.spectra])
        else:
            W = _gaussian_matrix(self.wavelengths, None, wl, np.broadcast_to(fwhm, wl.shape))
            W = W * np.gradient(self.wavelengths)[None, :]
            W /= np.maximum(W.sum(1, keepdims=True), 1e-12)
            S = np.nan_to_num(self.spectra) @ W.T
        return SpectralLibrary(wl, S, self.names, self.metadata)

    def plot(self, ax=None, **kw):
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(9, 4))
        for n, s in zip(self.names, self.spectra):
            ax.plot(self.wavelengths, s, label=n, **kw)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Reflectance")
        if len(self) <= 12:
            ax.legend(fontsize=8)
        return ax


def merge_libraries(libs: Sequence[SpectralLibrary], wavelengths: Sequence[float] | None = None
                    ) -> SpectralLibrary:
    """Combine libraries onto one wavelength grid (default: the first library's)."""
    wl = np.asarray(wavelengths if wavelengths is not None else libs[0].wavelengths, float)
    parts = [lib.resample(wl) for lib in libs]
    return SpectralLibrary(wl, np.vstack([p.spectra for p in parts]),
                           [n for p in parts for n in p.names])


def from_dict(spectra: Mapping[str, Sequence[float]], wavelengths: Sequence[float]
              ) -> SpectralLibrary:
    return SpectralLibrary(np.asarray(wavelengths, float), np.array(list(spectra.values()), float),
                           list(spectra))


def from_image(cube: xr.DataArray, points, label_col: str, *, buffer: int = 0,
               agg: str = "mean") -> SpectralLibrary:
    """Library of class spectra (mean or median) sampled at labelled points."""
    from .cube import spectra_at, wavelengths

    df = spectra_at(cube, points, label_col=label_col, buffer=buffer)
    g = getattr(df.groupby("label"), agg)()
    return SpectralLibrary(wavelengths(cube), g.to_numpy(float), [str(i) for i in g.index])


def read_csv(path: str | PathLike, *, wavelength_col: str | int = 0, units: str = "nm",
             scale: float = 1.0) -> SpectralLibrary:
    """CSV with a wavelength column and one column per spectrum."""
    df = pd.read_csv(path)
    wc = df.columns[wavelength_col] if isinstance(wavelength_col, int) else wavelength_col
    wl = df[wc].to_numpy(float) * (1000.0 if units in ("um", "µm") else 1.0)
    sp = df.drop(columns=[wc])
    return SpectralLibrary(wl, sp.to_numpy(float).T * scale, [str(c) for c in sp.columns])


def _read_numbers(path: Path) -> tuple[str, np.ndarray]:
    lines = path.read_text(errors="ignore").splitlines()
    vals, header = [], []
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        try:
            vals.append([float(v) for v in re.split(r"[\s,]+", s)])
        except ValueError:
            if not vals:
                header.append(s)
    return " ".join(header), np.array([v for v in vals if len(v) == len(vals[-1])], float)


def read_usgs(paths: str | PathLike | Sequence[str | PathLike], *,
              wavelength_file: str | PathLike | None = None) -> SpectralLibrary:
    """USGS Spectral Library v7 (splib07a/b ASCII). Each spectrum file has one header line
    and one value per line; wavelengths come from the matching ``*Wavelengths*`` file (µm).
    ``paths`` may be a folder, a glob pattern or a list of files. The 1.23e34 "deleted
    channel" marker becomes NaN."""
    files = _expand(paths)
    wfile = Path(wavelength_file) if wavelength_file else next(
        (f for f in files if "wavelength" in f.name.lower()), None)
    if wfile is None:
        raise ValueError("Pass wavelength_file= (the splib07 '*Wavelengths*' file)")
    _, wv = _read_numbers(wfile)
    wl = wv[:, 0] * (1000.0 if np.nanmax(wv[:, 0]) < 100 else 1.0)
    names, spectra = [], []
    for f in files:
        if f == wfile or "wavelength" in f.name.lower() or "bandpass" in f.name.lower():
            continue
        head, v = _read_numbers(f)
        if v.size == 0 or v.shape[0] != len(wl):
            continue
        s = v[:, 0].astype(float)
        s[s < -1e30] = np.nan
        s[np.abs(s) > 1e30] = np.nan
        names.append(head.replace("Record=", "").strip() or f.stem)
        spectra.append(s)
    return SpectralLibrary(wl, np.array(spectra), names)


def read_ecostress(paths: str | PathLike | Sequence[str | PathLike], *,
                   wavelengths: Sequence[float] | None = None) -> SpectralLibrary:
    """ECOSTRESS / ASTER spectral library text files (header ``Name:``, ``X Units:
    Wavelength (micrometers)``, ``Y Units: Reflectance (percent)``, then two columns).
    Spectra are put on ``wavelengths`` (nm; default: 400–2500 nm every 1 nm), and
    percent is converted to 0–1."""
    files = _expand(paths)
    grid = np.asarray(wavelengths if wavelengths is not None else np.arange(400, 2501), float)
    names, spectra, meta = [], [], []
    for f in files:
        text = f.read_text(errors="ignore").splitlines()
        info = {}
        for ln in text[:30]:
            if ":" in ln:
                k, v = ln.split(":", 1)
                info[k.strip().lower()] = v.strip()
        _, v = _read_numbers(f)
        if v.ndim != 2 or v.shape[1] < 2:
            continue
        x, y = v[:, 0], v[:, 1]
        if "micrometer" in info.get("x units", "micrometers").lower() or x.max() < 30:
            x = x * 1000.0
        if "percent" in info.get("y units", "percent").lower():
            y = y / 100.0
        o = np.argsort(x)
        spectra.append(np.interp(grid, x[o], y[o], left=np.nan, right=np.nan))
        names.append(info.get("name", f.stem))
        meta.append({"file": f.name, "class": info.get("class", ""),
                     "type": info.get("type", "")})
    return SpectralLibrary(grid, np.array(spectra), names, pd.DataFrame(meta))


def _expand(paths) -> list[Path]:
    if isinstance(paths, (str, PathLike)):
        p = Path(paths)
        if p.is_dir():
            return sorted(q for q in p.rglob("*") if q.suffix.lower() in (".txt", ".asc", ""))
        if any(ch in str(p) for ch in "*?["):
            return sorted(p.parent.glob(p.name))
        return [p]
    return [Path(q) for q in paths]


def synthetic_library(wavelengths: Sequence[float] | None = None) -> SpectralLibrary:
    """Idealised spectra (green vegetation, dry vegetation, soil, water, kaolinite-like clay,
    calcite-like carbonate) built from Gaussian absorption features. For teaching and tests
    only, not a substitute for measured libraries."""
    wl = np.asarray(wavelengths if wavelengths is not None else np.arange(400, 2501, 5), float)

    def g(c, w, d):
        return d * np.exp(-0.5 * ((wl - c) / w) ** 2)

    edge = 1 / (1 + np.exp(-(wl - 715) / 12))
    veg = 0.05 + 0.40 * edge - g(1450, 60, 0.18) - g(1940, 80, 0.22) - g(670, 25, 0.03) \
        + g(550, 30, 0.05) - g(980, 40, 0.05) - g(1200, 50, 0.07)
    veg = np.where(wl > 1300, veg * (1 - 0.25 * (wl - 1300) / 1200), veg)
    dry = 0.12 + 0.25 * (wl - 400) / 2100 + 0.08 * edge - g(2100, 60, 0.08) - g(1730, 40, 0.03) \
        - g(1450, 60, 0.04) - g(1940, 70, 0.05)
    soil = 0.08 + 0.30 * (1 - np.exp(-(wl - 380) / 700)) - g(1410, 40, 0.03) - g(1910, 50, 0.05) \
        - g(2200, 25, 0.03)
    water = 0.06 * np.exp(-(wl - 400) / 150) + 0.005
    clay = 0.55 + 0.1 * (1 - np.exp(-(wl - 400) / 300)) - g(1400, 15, 0.12) - g(2165, 12, 0.10) \
        - g(2205, 14, 0.18) - g(1910, 30, 0.04)
    carb = 0.6 + 0.08 * (1 - np.exp(-(wl - 400) / 300)) - g(2340, 25, 0.25) - g(1990, 30, 0.05) \
        - g(2160, 25, 0.06)
    S = np.clip(np.vstack([veg, dry, soil, water, clay, carb]), 0.001, 1)
    return SpectralLibrary(wl, S, ["green_vegetation", "dry_vegetation", "soil", "water",
                                   "clay_kaolinite", "carbonate_calcite"])


def as_library(obj: Any, cube: xr.DataArray | None = None) -> SpectralLibrary:
    """Coerce a SpectralLibrary / dict / DataFrame (rows = spectra) to a library on the cube's
    bands."""
    from .cube import wavelengths as _wl

    if isinstance(obj, SpectralLibrary):
        lib = obj
    elif isinstance(obj, Mapping):
        if cube is None:
            raise ValueError("Pass cube= to interpret a dict of spectra")
        lib = from_dict(obj, _wl(cube))
    elif isinstance(obj, pd.DataFrame):
        try:
            wl = obj.columns.to_numpy(float)
        except (TypeError, ValueError):
            wl = _wl(cube)
        lib = SpectralLibrary(wl, obj.to_numpy(float), [str(i) for i in obj.index])
    else:
        arr = np.atleast_2d(np.asarray(obj, float))
        lib = SpectralLibrary(_wl(cube), arr, [f"em{i + 1}" for i in range(len(arr))])
    if cube is not None and (len(lib.wavelengths) != cube.sizes["band"] or
                             not np.allclose(lib.wavelengths, _wl(cube))):
        lib = lib.resample(cube)
    return lib
