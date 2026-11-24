"""Narrowband (hyperspectral) indices defined by wavelength, so they work on any sensor:
pigments, red edge, water, dry matter/nitrogen, and mineral band ratios.

Each index is a function of ``R(nm)``, the reflectance of the band nearest to ``nm``
(within ``tolerance``). Add your own with :func:`register_index`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import xarray as xr

from .cube import band_at, wavelengths


@dataclass(frozen=True)
class NarrowbandIndex:
    name: str
    formula: Callable
    wavelengths: tuple[float, ...]
    description: str
    reference: str = ""


INDICES: dict[str, NarrowbandIndex] = {}


def register_index(name: str, wavelengths_nm: Sequence[float], formula: Callable,
                   description: str = "", reference: str = "") -> None:
    """Add an index: ``formula`` receives one reflectance array per wavelength, in order."""
    INDICES[name] = NarrowbandIndex(name, formula, tuple(float(w) for w in wavelengths_nm),
                                    description, reference)


def _nd(a, b):
    return (a - b) / (a + b)


_R = register_index
# vegetation structure and greenness
_R("NDVI", (800, 670), _nd, "Narrowband NDVI", "Rouse et al. 1974")
_R("EVI", (800, 670, 470), lambda n, r, b: 2.5 * (n - r) / (n + 6 * r - 7.5 * b + 1),
   "Enhanced vegetation index", "Huete et al. 2002")
_R("OSAVI", (800, 670), lambda n, r: 1.16 * (n - r) / (n + r + 0.16),
   "Optimized soil-adjusted VI", "Rondeaux et al. 1996")
_R("MTVI2", (800, 670, 550), lambda n, r, g: 1.5 * (1.2 * (n - g) - 2.5 * (r - g))
   / np.sqrt((2 * n + 1) ** 2 - (6 * n - 5 * np.sqrt(np.maximum(r, 0))) - 0.5),
   "Modified triangular VI 2 (LAI)", "Haboudane et al. 2004")
# chlorophyll / red edge
_R("NDRE", (790, 720), _nd, "Normalized difference red edge", "Barnes et al. 2000")
_R("mND705", (750, 705, 445), lambda a, b, c: (a - b) / (a + b - 2 * c),
   "Modified red-edge ND", "Sims & Gamon 2002")
_R("CIred_edge", (800, 720), lambda n, re: n / re - 1, "Red-edge chlorophyll index",
   "Gitelson et al. 2003")
_R("MTCI", (754, 709, 681), lambda a, b, c: (a - b) / (b - c), "MERIS terrestrial chlorophyll",
   "Dash & Curran 2004")
_R("MCARI", (700, 670, 550), lambda a, r, g: ((a - r) - 0.2 * (a - g)) * (a / r),
   "Modified chlorophyll absorption ratio", "Daughtry et al. 2000")
_R("TCARI", (700, 670, 550), lambda a, r, g: 3 * ((a - r) - 0.2 * (a - g) * (a / r)),
   "Transformed CARI", "Haboudane et al. 2002")
_R("TCARI_OSAVI", (700, 670, 550, 800),
   lambda a, r, g, n: 3 * ((a - r) - 0.2 * (a - g) * (a / r)) / (1.16 * (n - r) / (n + r + 0.16)),
   "TCARI/OSAVI (chlorophyll, LAI-insensitive)", "Haboudane et al. 2002")
_R("REP_linear", (670, 700, 740, 780),
   lambda a, b, c, d: 700 + 40 * (((a + d) / 2 - b) / (c - b)),
   "Red-edge position, 4-point linear interpolation (nm)", "Guyot & Baret 1988")
# pigments and light use
_R("PRI", (531, 570), _nd, "Photochemical reflectance index", "Gamon et al. 1992")
_R("CCI", (531, 645), _nd, "Chlorophyll/carotenoid index (MODIS-like)", "Gamon et al. 2016")
_R("SIPI", (800, 445, 680), lambda n, b, r: (n - b) / (n - r),
   "Structure-insensitive pigment index", "Penuelas et al. 1995")
_R("PSRI", (680, 500, 750), lambda r, b, n: (r - b) / n, "Plant senescence reflectance",
   "Merzlyak et al. 1999")
_R("CRI1", (510, 550), lambda a, b: 1 / a - 1 / b, "Carotenoid reflectance index 1",
   "Gitelson et al. 2002")
_R("ARI1", (550, 700), lambda a, b: 1 / a - 1 / b, "Anthocyanin reflectance index 1",
   "Gitelson et al. 2001")
# water
_R("WBI", (900, 970), lambda a, b: a / b, "Water band index", "Penuelas et al. 1997")
_R("NDWI_Gao", (860, 1240), _nd, "Leaf water (Gao 1996)", "Gao 1996")
_R("NDII", (820, 1650), _nd, "Normalized difference infrared index", "Hardisky et al. 1983")
_R("MSI", (1600, 820), lambda a, b: a / b, "Moisture stress index", "Hunt & Rock 1989")
# dry matter, lignin, nitrogen
_R("CAI", (2000, 2100, 2200), lambda a, b, c: 0.5 * (a + c) - b, "Cellulose absorption index",
   "Nagler et al. 2003")
_R("NDLI", (1754, 1680), lambda a, b: _nd(np.log(1 / a), np.log(1 / b)),
   "Normalized difference lignin index", "Serrano et al. 2002")
_R("NDNI", (1510, 1680), lambda a, b: _nd(np.log(1 / a), np.log(1 / b)),
   "Normalized difference nitrogen index", "Serrano et al. 2002")
# soils and minerals (band ratios / simple depths; use band_depth for full feature analysis)
_R("ClayAlOH", (2165, 2205, 2260), lambda a, b, c: (a + c) / (2 * b),
   "Al-OH (clay, ~2200 nm) relative depth", "Mars & Rowan 2006")
_R("Carbonate", (2295, 2335, 2385), lambda a, b, c: (a + c) / (2 * b),
   "Carbonate (~2335 nm) relative depth", "Mars & Rowan 2006")
_R("FerricIron", (750, 560), lambda a, b: a / b, "Ferric iron (red/green slope)",
   "Rowan et al. 1977")
_R("FerrousIron", (1650, 1000), lambda a, b: a / b, "Ferrous iron (~1000 nm absorption)", "")
_R("NBR_hs", (860, 2200), _nd, "Normalized burn ratio", "Key & Benson 2006")
_R("PhycocyaninRatio", (709, 620), lambda a, b: a / b, "Phycocyanin (cyanobacteria) ratio",
   "Simis et al. 2005")
_R("Chl_NDCI", (708, 665), _nd, "Normalized difference chlorophyll index (water)",
   "Mishra & Mishra 2012")


def list_indices() -> list[str]:
    return sorted(INDICES)


def available(cube: xr.DataArray, *, tolerance: float = 15.0) -> list[str]:
    """Indices whose wavelengths all have a band within ``tolerance`` nm."""
    wl = wavelengths(cube)
    return sorted(n for n, ix in INDICES.items()
                  if all(np.min(np.abs(wl - w)) <= tolerance for w in ix.wavelengths))


def index(cube: xr.DataArray, name: str | Sequence[str], *, tolerance: float = 15.0,
          width: float | None = None) -> xr.DataArray | xr.Dataset:
    """Compute one index (DataArray) or several (Dataset). ``width`` averages bands over
    ``±width/2`` nm around each wavelength (useful for noisy, finely sampled data)."""
    if not isinstance(name, str):
        return xr.Dataset({n: index(cube, n, tolerance=tolerance, width=width) for n in name})
    ix = INDICES[name]
    bands = [band_at(cube, w, tolerance=tolerance, width=width).astype("float64")
             for w in ix.wavelengths]
    with np.errstate(divide="ignore", invalid="ignore"):
        out = ix.formula(*bands)
    out = out.where(np.isfinite(out))
    return out.rename(name).assign_attrs(description=ix.description, reference=ix.reference,
                                         wavelengths=list(ix.wavelengths))


def red_edge_position(cube: xr.DataArray, *, method: str = "max_derivative",
                      start: float = 680.0, stop: float = 760.0) -> xr.DataArray:
    """Red-edge inflection point (nm): wavelength of the maximum first derivative between
    ``start`` and ``stop`` (refined by a parabola through the peak), or the 4-point
    ``linear`` method."""
    if method == "linear":
        return index(cube, "REP_linear").rename("rep")
    from .cube import sel_range

    c = sel_range(cube, start - 10, stop + 10)
    wl = wavelengths(c)
    a = np.asarray(c.values, dtype="float64")
    d = np.gradient(a, wl, axis=0)
    inside = (wl >= start) & (wl <= stop)
    d_in = np.where(inside[:, None, None], d, -np.inf)
    j = np.argmax(np.nan_to_num(d_in, nan=-np.inf), axis=0)
    jm, jp = np.clip(j - 1, 0, len(wl) - 1), np.clip(j + 1, 0, len(wl) - 1)
    take = np.take_along_axis
    y0, y1, y2 = (take(d, k[None], 0)[0] for k in (jm, j, jp))
    den = y0 - 2 * y1 + y2
    off = np.where(np.abs(den) > 1e-12, 0.5 * (y0 - y2) / den, 0.0)
    step = np.median(np.diff(wl))
    rep = wl[j] + np.clip(off, -1, 1) * step
    rep = np.where(np.isfinite(a).all(0), rep, np.nan)
    tpl = c.isel(band=0, drop=True)
    tpl = tpl.drop_vars([v for v in ("wavelength", "fwhm", "good") if v in tpl.coords])
    return tpl.copy(data=rep.astype("float32")).rename("rep").assign_attrs(units="nm")
