"""Spectral index registry.

Each index is declared once with canonical band names and a formula. The same
formula runs on NumPy, CuPy or Dask data because it uses only arithmetic and
NumPy ufuncs, which dispatch to the array type underneath the xarray object.

Add your own with :func:`register_index`. Which of these a given satellite can
actually compute is decided in :mod:`geolattice.rsense.spectral_index`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field

import numpy as np
import xarray as xr

from ..backend import get_backend, to_device
from .bands import get_band


@dataclass(frozen=True)
class SpectralIndex:
    name: str
    long_name: str
    bands: tuple[str, ...]
    formula: Callable[..., xr.DataArray]
    domain: str
    reference: str
    params: Mapping[str, float] = field(default_factory=dict)


INDICES: dict[str, SpectralIndex] = {}


def register_index(name: str, long_name: str, bands: Iterable[str], formula: Callable,
                   *, domain: str = "other", reference: str = "",
                   params: Mapping[str, float] | None = None, overwrite: bool = False) -> None:
    """Register a spectral index.

    ``formula`` receives each band as a keyword argument (plus any ``params``)::

        rs.optical.register_index(
            "SR", "Simple Ratio", ["nir", "red"], lambda nir, red: nir / red,
            domain="vegetation")
    """
    key = name.upper()
    if key in INDICES and not overwrite:
        raise ValueError(f"Index {name!r} already registered; pass overwrite=True to replace")
    INDICES[key] = SpectralIndex(name, long_name, tuple(bands), formula, domain, reference,
                                 dict(params or {}))


def _nd(a, b):
    return (a - b) / (a + b)


_R = register_index
# --- vegetation
_R("NDVI", "Normalized Difference Vegetation Index", ["nir", "red"],
   lambda nir, red: _nd(nir, red), domain="vegetation", reference="Rouse et al. 1974")
_R("EVI", "Enhanced Vegetation Index", ["nir", "red", "blue"],
   lambda nir, red, blue, G, C1, C2, L: G * (nir - red) / (nir + C1 * red - C2 * blue + L),
   domain="vegetation", reference="Huete et al. 2002",
   params={"G": 2.5, "C1": 6.0, "C2": 7.5, "L": 1.0})
_R("EVI2", "Two-band Enhanced Vegetation Index", ["nir", "red"],
   lambda nir, red: 2.5 * (nir - red) / (nir + 2.4 * red + 1.0),
   domain="vegetation", reference="Jiang et al. 2008")
_R("SAVI", "Soil-Adjusted Vegetation Index", ["nir", "red"],
   lambda nir, red, L: (1 + L) * (nir - red) / (nir + red + L),
   domain="vegetation", reference="Huete 1988", params={"L": 0.5})
_R("OSAVI", "Optimized Soil-Adjusted Vegetation Index", ["nir", "red"],
   lambda nir, red: (nir - red) / (nir + red + 0.16),
   domain="vegetation", reference="Rondeaux et al. 1996")
_R("MSAVI", "Modified Soil-Adjusted Vegetation Index", ["nir", "red"],
   lambda nir, red: (2 * nir + 1 - np.sqrt((2 * nir + 1) ** 2 - 8 * (nir - red))) / 2,
   domain="vegetation", reference="Qi et al. 1994")
_R("GNDVI", "Green NDVI", ["nir", "green"], lambda nir, green: _nd(nir, green),
   domain="vegetation", reference="Gitelson et al. 1996")
_R("ARVI", "Atmospherically Resistant Vegetation Index", ["nir", "red", "blue"],
   lambda nir, red, blue, gamma: _nd(nir, red - gamma * (blue - red)),
   domain="vegetation", reference="Kaufman & Tanre 1992", params={"gamma": 1.0})
_R("kNDVI", "Kernel NDVI", ["nir", "red"], lambda nir, red: np.tanh(_nd(nir, red) ** 2),
   domain="vegetation", reference="Camps-Valls et al. 2021")
_R("NDRE", "Normalized Difference Red Edge", ["nir", "rededge1"],
   lambda nir, rededge1: _nd(nir, rededge1), domain="vegetation", reference="Barnes et al. 2000")
_R("CIre", "Red-edge Chlorophyll Index", ["nir", "rededge1"],
   lambda nir, rededge1: nir / rededge1 - 1, domain="vegetation", reference="Gitelson et al. 2003")
_R("CIg", "Green Chlorophyll Index", ["nir", "green"], lambda nir, green: nir / green - 1,
   domain="vegetation", reference="Gitelson et al. 2003")
_R("OTCI", "OLCI Terrestrial Chlorophyll Index", ["oa12", "oa11", "oa10"],
   lambda oa12, oa11, oa10: (oa12 - oa11) / (oa11 - oa10), domain="vegetation",
   reference="Dash & Curran 2004 (as MTCI); Sentinel-3 OLCI bands 10/11/12, 681/709/754 nm")
# --- water
_R("NDWI", "Normalized Difference Water Index", ["green", "nir"],
   lambda green, nir: _nd(green, nir), domain="water", reference="McFeeters 1996")
_R("MNDWI", "Modified NDWI", ["green", "swir16"], lambda green, swir16: _nd(green, swir16),
   domain="water", reference="Xu 2006")
_R("AWEInsh", "Automated Water Extraction Index (no shadow)", ["green", "nir", "swir16", "swir22"],
   lambda green, nir, swir16, swir22: 4 * (green - swir16) - (0.25 * nir + 2.75 * swir22),
   domain="water", reference="Feyisa et al. 2014")
_R("AWEIsh", "Automated Water Extraction Index (shadow)",
   ["blue", "green", "nir", "swir16", "swir22"],
   lambda blue, green, nir, swir16, swir22: blue + 2.5 * green - 1.5 * (nir + swir16) - 0.25 * swir22,
   domain="water", reference="Feyisa et al. 2014")
_R("NDCI", "Normalized Difference Chlorophyll Index", ["rededge1", "red"],
   lambda rededge1, red: _nd(rededge1, red), domain="water", reference="Mishra & Mishra 2012")
_R("MCI", "Maximum Chlorophyll Index", ["oa11", "oa10", "oa12"],
   lambda oa11, oa10, oa12: oa11 - oa10 - (oa12 - oa10) * (708.75 - 681.25) / (753.75 - 681.25),
   domain="water", reference="Gower et al. 2005; Sentinel-3 OLCI bands 10/11/12, 681/709/754 nm")
_R("NDMI", "Normalized Difference Moisture Index", ["nir", "swir16"],
   lambda nir, swir16: _nd(nir, swir16), domain="water", reference="Gao 1996")
# --- fire / burn
_R("NBR", "Normalized Burn Ratio", ["nir", "swir22"], lambda nir, swir22: _nd(nir, swir22),
   domain="burn", reference="Key & Benson 2006")
_R("NBR2", "Normalized Burn Ratio 2", ["swir16", "swir22"],
   lambda swir16, swir22: _nd(swir16, swir22), domain="burn", reference="USGS")
# --- urban / soil / snow
_R("NDBI", "Normalized Difference Built-up Index", ["swir16", "nir"],
   lambda swir16, nir: _nd(swir16, nir), domain="urban", reference="Zha et al. 2003")
_R("BSI", "Bare Soil Index", ["swir16", "red", "nir", "blue"],
   lambda swir16, red, nir, blue: _nd(swir16 + red, nir + blue),
   domain="soil", reference="Rikimaru et al. 2002")
_R("NDSI", "Normalized Difference Snow Index", ["green", "swir16"],
   lambda green, swir16: _nd(green, swir16), domain="snow", reference="Hall et al. 1995")

# ============================================================================ extended catalog
# --- vegetation: broadband
_R("SR", "Simple Ratio", ["nir", "red"], lambda nir, red: nir / red,
   domain="vegetation", reference="Jordan 1969")
_R("DVI", "Difference Vegetation Index", ["nir", "red"], lambda nir, red: nir - red,
   domain="vegetation", reference="Tucker 1979")
_R("IPVI", "Infrared Percentage Vegetation Index", ["nir", "red"],
   lambda nir, red: nir / (nir + red), domain="vegetation", reference="Crippen 1990")
_R("RDVI", "Renormalized Difference Vegetation Index", ["nir", "red"],
   lambda nir, red: (nir - red) / np.sqrt(nir + red), domain="vegetation",
   reference="Roujean & Breon 1995")
_R("MSR", "Modified Simple Ratio", ["nir", "red"],
   lambda nir, red: (nir / red - 1) / np.sqrt(nir / red + 1), domain="vegetation",
   reference="Chen 1996")
_R("NLI", "Non-Linear Index", ["nir", "red"], lambda nir, red: (nir**2 - red) / (nir**2 + red),
   domain="vegetation", reference="Goel & Qin 1994")
_R("MNLI", "Modified Non-Linear Index", ["nir", "red"],
   lambda nir, red, L: (1 + L) * (nir**2 - red) / (nir**2 + red + L), domain="vegetation",
   reference="Yang et al. 2008", params={"L": 0.5})
_R("TDVI", "Transformed Difference Vegetation Index", ["nir", "red"],
   lambda nir, red: 1.5 * (nir - red) / np.sqrt(nir**2 + red + 0.5), domain="vegetation",
   reference="Bannari et al. 2002")
_R("TNDVI", "Transformed NDVI", ["nir", "red"], lambda nir, red: np.sqrt(_nd(nir, red) + 0.5),
   domain="vegetation", reference="Tucker 1979")
_R("WDRVI", "Wide Dynamic Range Vegetation Index", ["nir", "red"],
   lambda nir, red, alpha: (alpha * nir - red) / (alpha * nir + red), domain="vegetation",
   reference="Gitelson 2004", params={"alpha": 0.2})
_R("GARI", "Green Atmospherically Resistant Index", ["nir", "green", "blue", "red"],
   lambda nir, green, blue, red, gamma: _nd(nir, green - gamma * (blue - red)),
   domain="vegetation", reference="Gitelson et al. 1996", params={"gamma": 1.7})
_R("GEMI", "Global Environment Monitoring Index", ["nir", "red"],
   lambda nir, red: (lambda eta: eta * (1 - 0.25 * eta) - (red - 0.125) / (1 - red))(
       (2 * (nir**2 - red**2) + 1.5 * nir + 0.5 * red) / (nir + red + 0.5)),
   domain="vegetation", reference="Pinty & Verstraete 1992")
_R("NIRv", "Near-Infrared Reflectance of Vegetation", ["nir", "red"],
   lambda nir, red: _nd(nir, red) * nir, domain="vegetation", reference="Badgley et al. 2017")
_R("GSAVI", "Green Soil-Adjusted Vegetation Index", ["nir", "green"],
   lambda nir, green, L: (1 + L) * (nir - green) / (nir + green + L), domain="vegetation",
   reference="Sripada et al. 2006", params={"L": 0.5})
_R("GOSAVI", "Green Optimized Soil-Adjusted Vegetation Index", ["nir", "green"],
   lambda nir, green: (nir - green) / (nir + green + 0.16), domain="vegetation",
   reference="Sripada et al. 2006")
_R("BNDVI", "Blue NDVI", ["nir", "blue"], lambda nir, blue: _nd(nir, blue),
   domain="vegetation", reference="Hancock & Dougherty 2007")
_R("BWDRVI", "Blue Wide Dynamic Range Vegetation Index", ["nir", "blue"],
   lambda nir, blue: (0.1 * nir - blue) / (0.1 * nir + blue), domain="vegetation",
   reference="Hancock & Dougherty 2007")
_R("GRNDVI", "Green-Red NDVI", ["nir", "green", "red"],
   lambda nir, green, red: (nir - (green + red)) / (nir + green + red), domain="vegetation",
   reference="Wang et al. 2007")
_R("GRVI", "Green Ratio Vegetation Index", ["nir", "green"], lambda nir, green: nir / green,
   domain="vegetation", reference="Sripada et al. 2006")
_R("CVI", "Chlorophyll Vegetation Index", ["nir", "red", "green"],
   lambda nir, red, green: nir * red / green**2, domain="vegetation",
   reference="Vincini et al. 2008")
_R("AVI", "Advanced Vegetation Index", ["nir", "red"],
   lambda nir, red: np.cbrt(nir * (1 - red) * (nir - red)), domain="vegetation",
   reference="Rikimaru et al. 2002")
_R("SLAVI", "Specific Leaf Area Vegetation Index", ["nir", "red", "swir22"],
   lambda nir, red, swir22: nir / (red + swir22), domain="vegetation", reference="Lymburner 2000")
_R("SIPI", "Structure Insensitive Pigment Index", ["nir", "blue", "red"],
   lambda nir, blue, red: (nir - blue) / (nir - red), domain="vegetation",
   reference="Penuelas et al. 1995")
_R("MCARI1", "Modified Chlorophyll Absorption Ratio Index 1", ["nir", "red", "green"],
   lambda nir, red, green: 1.2 * (2.5 * (nir - red) - 1.3 * (nir - green)),
   domain="vegetation", reference="Haboudane et al. 2004")
_R("MTVI1", "Modified Triangular Vegetation Index 1", ["nir", "green", "red"],
   lambda nir, green, red: 1.2 * (1.2 * (nir - green) - 2.5 * (red - green)),
   domain="vegetation", reference="Haboudane et al. 2004")
_R("MTVI2", "Modified Triangular Vegetation Index 2", ["nir", "green", "red"],
   lambda nir, green, red: 1.5 * (1.2 * (nir - green) - 2.5 * (red - green))
   / np.sqrt((2 * nir + 1) ** 2 - (6 * nir - 5 * np.sqrt(red)) - 0.5),
   domain="vegetation", reference="Haboudane et al. 2004")
_R("TVI", "Triangular Vegetation Index", ["nir", "green", "red"],
   lambda nir, green, red: 0.5 * (120 * (nir - green) - 200 * (red - green)),
   domain="vegetation", reference="Broge & Leblanc 2001")
_R("NDDI", "Normalized Difference Drought Index", ["nir", "red", "green"],
   lambda nir, red, green: _nd(_nd(nir, red), _nd(green, nir)), domain="vegetation",
   reference="Gu et al. 2007")
# --- vegetation: visible / RGB (UAV and phenocam)
_R("VARI", "Visible Atmospherically Resistant Index", ["green", "red", "blue"],
   lambda green, red, blue: (green - red) / (green + red - blue), domain="rgb",
   reference="Gitelson et al. 2002")
_R("GLI", "Green Leaf Index", ["green", "red", "blue"],
   lambda green, red, blue: (2 * green - red - blue) / (2 * green + red + blue), domain="rgb",
   reference="Louhaichi et al. 2001")
_R("ExG", "Excess Green", ["green", "red", "blue"],
   lambda green, red, blue: 2 * green - red - blue, domain="rgb", reference="Woebbecke et al. 1995")
_R("ExR", "Excess Red", ["red", "green"], lambda red, green: 1.3 * red - green, domain="rgb",
   reference="Meyer et al. 1998")
_R("ExGR", "Excess Green minus Excess Red", ["green", "red", "blue"],
   lambda green, red, blue: (2 * green - red - blue) - (1.3 * red - green), domain="rgb",
   reference="Meyer & Neto 2008")
_R("NGRDI", "Normalized Green-Red Difference Index", ["green", "red"],
   lambda green, red: _nd(green, red), domain="rgb", reference="Tucker 1979")
_R("TGI", "Triangular Greenness Index", ["red", "green", "blue"],
   lambda red, green, blue: -0.5 * (190 * (red - green) - 120 * (red - blue)), domain="rgb",
   reference="Hunt et al. 2013")
_R("RGBVI", "Red Green Blue Vegetation Index", ["green", "blue", "red"],
   lambda green, blue, red: (green**2 - blue * red) / (green**2 + blue * red), domain="rgb",
   reference="Bendig et al. 2015")
_R("MGRVI", "Modified Green Red Vegetation Index", ["green", "red"],
   lambda green, red: (green**2 - red**2) / (green**2 + red**2), domain="rgb",
   reference="Bendig et al. 2015")
_R("GCC", "Green Chromatic Coordinate", ["green", "red", "blue"],
   lambda green, red, blue: green / (red + green + blue), domain="rgb",
   reference="Richardson et al. 2007")
_R("RCC", "Red Chromatic Coordinate", ["red", "green", "blue"],
   lambda red, green, blue: red / (red + green + blue), domain="rgb",
   reference="Richardson et al. 2007")
_R("BCC", "Blue Chromatic Coordinate", ["blue", "red", "green"],
   lambda blue, red, green: blue / (red + green + blue), domain="rgb",
   reference="Richardson et al. 2007")
_R("NDYI", "Normalized Difference Yellowness Index", ["green", "blue"],
   lambda green, blue: _nd(green, blue), domain="rgb", reference="Sulik & Long 2016")
_R("SI", "Shadow Index", ["blue", "green", "red"],
   lambda blue, green, red: np.cbrt((1 - blue) * (1 - green) * (1 - red)), domain="rgb",
   reference="Rikimaru et al. 2002")
# --- vegetation: red edge and pigments (Sentinel-2, multispectral UAV)
_R("NDRE2", "Normalized Difference Red Edge (re2)", ["nir", "rededge2"],
   lambda nir, rededge2: _nd(nir, rededge2), domain="rededge", reference="Barnes et al. 2000")
_R("ND705", "Red-edge NDVI (705/740 nm)", ["rededge2", "rededge1"],
   lambda rededge2, rededge1: _nd(rededge2, rededge1), domain="rededge",
   reference="Gitelson & Merzlyak 1994")
_R("MSR705", "Modified Simple Ratio (705/740 nm)", ["rededge2", "rededge1"],
   lambda rededge2, rededge1: (rededge2 / rededge1 - 1) / np.sqrt(rededge2 / rededge1 + 1),
   domain="rededge", reference="Wu et al. 2008")
_R("MTCI", "MERIS Terrestrial Chlorophyll Index", ["rededge2", "rededge1", "red"],
   lambda rededge2, rededge1, red: (rededge2 - rededge1) / (rededge1 - red), domain="rededge",
   reference="Dash & Curran 2004")
_R("IRECI", "Inverted Red-Edge Chlorophyll Index", ["rededge3", "red", "rededge1", "rededge2"],
   lambda rededge3, red, rededge1, rededge2: (rededge3 - red) / (rededge1 / rededge2),
   domain="rededge", reference="Frampton et al. 2013")
_R("S2REP", "Sentinel-2 Red-Edge Position (nm)", ["rededge3", "red", "rededge1", "rededge2"],
   lambda rededge3, red, rededge1, rededge2:
   705 + 35 * (((rededge3 + red) / 2 - rededge1) / (rededge2 - rededge1)),
   domain="rededge", reference="Frampton et al. 2013")
_R("REIP", "Red-Edge Inflection Point (nm)", ["red", "rededge3", "rededge1", "rededge2"],
   lambda red, rededge3, rededge1, rededge2:
   700 + 40 * (((red + rededge3) / 2 - rededge1) / (rededge2 - rededge1)),
   domain="rededge", reference="Guyot & Baret 1988")
_R("CCCI", "Canopy Chlorophyll Content Index", ["nir", "rededge1", "red"],
   lambda nir, rededge1, red: _nd(nir, rededge1) / _nd(nir, red), domain="rededge",
   reference="Barnes et al. 2000")
_R("MCARI", "Modified Chlorophyll Absorption in Reflectance Index",
   ["rededge1", "red", "green"],
   lambda rededge1, red, green: ((rededge1 - red) - 0.2 * (rededge1 - green)) * (rededge1 / red),
   domain="rededge", reference="Daughtry et al. 2000")
_R("TCARI", "Transformed Chlorophyll Absorption in Reflectance Index",
   ["rededge1", "red", "green"],
   lambda rededge1, red, green: 3 * ((rededge1 - red) - 0.2 * (rededge1 - green) * (rededge1 / red)),
   domain="rededge", reference="Haboudane et al. 2002")
_R("TCARI_OSAVI", "TCARI/OSAVI ratio", ["rededge1", "red", "green", "nir"],
   lambda rededge1, red, green, nir:
   3 * ((rededge1 - red) - 0.2 * (rededge1 - green) * (rededge1 / red))
   / ((nir - red) / (nir + red + 0.16)), domain="rededge", reference="Haboudane et al. 2002")
_R("MCARI_OSAVI", "MCARI/OSAVI ratio", ["rededge1", "red", "green", "nir"],
   lambda rededge1, red, green, nir:
   ((rededge1 - red) - 0.2 * (rededge1 - green)) * (rededge1 / red)
   / ((nir - red) / (nir + red + 0.16)), domain="rededge", reference="Daughtry et al. 2000")
_R("PSRI", "Plant Senescence Reflectance Index", ["red", "blue", "rededge2"],
   lambda red, blue, rededge2: (red - blue) / rededge2, domain="rededge",
   reference="Merzlyak et al. 1999")
_R("ARI", "Anthocyanin Reflectance Index", ["green", "rededge1"],
   lambda green, rededge1: 1 / green - 1 / rededge1, domain="rededge",
   reference="Gitelson et al. 2001")
_R("MARI", "Modified Anthocyanin Reflectance Index", ["green", "rededge1", "nir"],
   lambda green, rededge1, nir: (1 / green - 1 / rededge1) * nir, domain="rededge",
   reference="Gitelson et al. 2006")
_R("CRI550", "Carotenoid Reflectance Index 550", ["blue", "green"],
   lambda blue, green: 1 / blue - 1 / green, domain="rededge", reference="Gitelson et al. 2002")
_R("CRI700", "Carotenoid Reflectance Index 700", ["blue", "rededge1"],
   lambda blue, rededge1: 1 / blue - 1 / rededge1, domain="rededge",
   reference="Gitelson et al. 2002")
_R("REDSI", "Red-Edge Disease Stress Index", ["rededge3", "red", "rededge1"],
   lambda rededge3, red, rededge1: ((705 - 665) * (rededge3 - red) - (783 - 665) * (rededge1 - red))
   / (2 * red), domain="rededge", reference="Zheng et al. 2018")
# --- water and moisture
_R("WI2015", "Water Index 2015", ["green", "red", "nir", "swir16", "swir22"],
   lambda green, red, nir, swir16, swir22:
   1.7204 + 171 * green + 3 * red - 70 * nir - 45 * swir16 - 71 * swir22,
   domain="water", reference="Fisher et al. 2016")
_R("WRI", "Water Ratio Index", ["green", "red", "nir", "swir16"],
   lambda green, red, nir, swir16: (green + red) / (nir + swir16), domain="water",
   reference="Shen & Li 2010")
_R("NWI", "New Water Index", ["blue", "nir", "swir16", "swir22"],
   lambda blue, nir, swir16, swir22: _nd(blue, nir + swir16 + swir22), domain="water",
   reference="Ding 2009")
_R("MBWI", "Multi-Band Water Index", ["green", "red", "nir", "swir16", "swir22"],
   lambda green, red, nir, swir16, swir22: 2 * green - red - nir - swir16 - swir22,
   domain="water", reference="Wang et al. 2018")
_R("SWM", "Sentinel Water Mask", ["blue", "green", "nir", "swir16"],
   lambda blue, green, nir, swir16: (blue + green) / (nir + swir16), domain="water",
   reference="Milczarek et al. 2017")
_R("ANDWI", "Augmented NDWI", ["blue", "green", "red", "nir", "swir16", "swir22"],
   lambda blue, green, red, nir, swir16, swir22: _nd(blue + green + red, nir + swir16 + swir22),
   domain="water", reference="Rad et al. 2021")
_R("MuWIR", "Revised Multi-Spectral Water Index", ["blue", "green", "nir", "swir16", "swir22"],
   lambda blue, green, nir, swir16, swir22: -4 * _nd(blue, green) + 2 * _nd(green, nir)
   + 2 * _nd(green, swir22) - _nd(green, swir16), domain="water", reference="Wang et al. 2018")
_R("S2WI", "Sentinel-2 Water Index", ["rededge1", "swir22"],
   lambda rededge1, swir22: _nd(rededge1, swir22), domain="water", reference="Jiang et al. 2020")
_R("NDTI", "Normalized Difference Turbidity Index", ["red", "green"],
   lambda red, green: _nd(red, green), domain="water", reference="Lacaux et al. 2007")
_R("NDPI", "Normalized Difference Pond Index", ["swir16", "green"],
   lambda swir16, green: _nd(swir16, green), domain="water", reference="Lacaux et al. 2007")
_R("MSI", "Moisture Stress Index", ["swir16", "nir"], lambda swir16, nir: swir16 / nir,
   domain="water", reference="Hunt & Rock 1989")
_R("NMDI", "Normalized Multi-band Drought Index", ["nir", "swir16", "swir22"],
   lambda nir, swir16, swir22: _nd(nir, swir16 - swir22), domain="water",
   reference="Wang & Qu 2007")
_R("GVMI", "Global Vegetation Moisture Index", ["nir", "swir22"],
   lambda nir, swir22: _nd(nir + 0.1, swir22 + 0.02), domain="water",
   reference="Ceccato et al. 2002")
# --- fire / burn
_R("BAI", "Burned Area Index", ["red", "nir"],
   lambda red, nir: 1 / ((0.1 - red) ** 2 + (0.06 - nir) ** 2), domain="burn",
   reference="Chuvieco et al. 2002")
_R("BAIM", "Burned Area Index (MODIS)", ["nir", "swir22"],
   lambda nir, swir22: 1 / ((0.05 - nir) ** 2 + (0.2 - swir22) ** 2), domain="burn",
   reference="Martin et al. 2005")
_R("MIRBI", "Mid-Infrared Burn Index", ["swir22", "swir16"],
   lambda swir22, swir16: 10 * swir22 - 9.8 * swir16 + 2, domain="burn",
   reference="Trigg & Flasse 2001")
_R("NBRSWIR", "Normalized Burn Ratio SWIR", ["swir22", "swir16"],
   lambda swir22, swir16: (swir22 - swir16 - 0.02) / (swir22 + swir16 + 0.1), domain="burn",
   reference="Liu et al. 2020")
_R("CSI", "Char Soil Index", ["nir", "swir22"], lambda nir, swir22: nir / swir22,
   domain="burn", reference="Smith et al. 2005")
_R("NBRplus", "Normalized Burn Ratio Plus", ["swir22", "nir08", "green", "blue"],
   lambda swir22, nir08, green, blue: _nd(swir22, nir08 + green + blue),
   domain="burn", reference="Alcaras et al. 2022")
_R("BAIS2", "Burned Area Index for Sentinel-2",
   ["rededge2", "rededge3", "nir08", "red", "swir22"],
   lambda rededge2, rededge3, nir08, red, swir22:
   (1 - np.sqrt(rededge2 * rededge3 * nir08 / red))
   * ((swir22 - nir08) / np.sqrt(swir22 + nir08) + 1), domain="burn",
   reference="Filipponi 2018")
# --- urban, soil, snow
_R("UI", "Urban Index", ["swir22", "nir"], lambda swir22, nir: _nd(swir22, nir), domain="urban",
   reference="Kawamura et al. 1996")
_R("IBI", "Index-based Built-up Index", ["swir16", "nir", "red", "green"],
   lambda swir16, nir, red, green, L:
   (lambda ndbi, savi, mndwi: (ndbi - (savi + mndwi) / 2) / (ndbi + (savi + mndwi) / 2))(
       _nd(swir16, nir), (1 + L) * (nir - red) / (nir + red + L), _nd(green, swir16)),
   domain="urban", reference="Xu 2008", params={"L": 0.5})
_R("EBBI", "Enhanced Built-up and Bareness Index", ["swir16", "nir", "thermal"],
   lambda swir16, nir, thermal: (swir16 - nir) / (10 * np.sqrt(swir16 + thermal)),
   domain="urban", reference="As-syakur et al. 2012")
_R("NBAI", "Normalized Built-up Area Index", ["swir22", "swir16", "green"],
   lambda swir22, swir16, green: (swir22 - swir16 / green) / (swir22 + swir16 / green),
   domain="urban", reference="Waqar et al. 2012")
_R("BLFEI", "Built-up Land Features Extraction Index", ["green", "red", "swir22", "swir16"],
   lambda green, red, swir22, swir16: _nd((green + red + swir22) / 3, swir16), domain="urban",
   reference="Bouhennache et al. 2019")
_R("NDBaI", "Normalized Difference Bareness Index", ["swir16", "thermal"],
   lambda swir16, thermal: _nd(swir16, thermal), domain="soil", reference="Zhao & Chen 2005")
_R("DBSI", "Dry Bareness Index", ["swir16", "green", "nir", "red"],
   lambda swir16, green, nir, red: _nd(swir16, green) - _nd(nir, red), domain="soil",
   reference="Rasul et al. 2018")
_R("NDSoI", "Normalized Difference Soil Index", ["swir22", "green"],
   lambda swir22, green: _nd(swir22, green), domain="soil", reference="Rogers & Kearney 2004")
_R("BareI", "Bareness Index", ["red", "swir16", "nir"],
   lambda red, swir16, nir: red + swir16 - nir, domain="soil", reference="Zhao & Chen 2005")
_R("BRBA", "Band Ratio for Built-up Area", ["red", "swir16"],
   lambda red, swir16: red / swir16, domain="urban", reference="Waqar et al. 2012")
_R("S3", "S3 Snow Index", ["nir", "red", "swir16"],
   lambda nir, red, swir16: nir * (red - swir16) / ((nir + red) * (nir + swir16)),
   domain="snow", reference="Saito & Yamazaki 1999")
_R("SWI", "Snow Water Index", ["green", "nir", "swir16"],
   lambda green, nir, swir16: green * (nir - swir16) / ((green + nir) * (nir + swir16)),
   domain="snow", reference="Dixit et al. 2019")
_R("NDGlaI", "Normalized Difference Glacier Index", ["green", "red"],
   lambda green, red: _nd(green, red), domain="snow", reference="Keshri et al. 2009")
# --- normalized band coordinates (useful ML features)
_R("NormG", "Normalized Green", ["green", "nir", "red"],
   lambda green, nir, red: green / (nir + green + red), domain="other", reference="Sripada 2006")
_R("NormNIR", "Normalized NIR", ["nir", "green", "red"],
   lambda nir, green, red: nir / (nir + green + red), domain="other", reference="Sripada 2006")
_R("NormR", "Normalized Red", ["red", "nir", "green"],
   lambda red, nir, green: red / (nir + green + red), domain="other", reference="Sripada 2006")
del _R

# Aliases: alternative names for the same formula
ALIASES = {"NDII": "NDMI", "LSWI": "NDMI", "RENDVI": "ND705", "MSAVI2": "MSAVI",
           "NDSI_SNOW": "NDSI", "NGBDI": "NDYI", "SR2": "GRVI", "VIG": "NGRDI"}


def list_indices(domain: str | None = None):
    """Return a DataFrame of registered indices, optionally filtered by domain."""
    import pandas as pd

    rows = [{"name": i.name, "long_name": i.long_name, "bands": ", ".join(i.bands),
             "domain": i.domain, "reference": i.reference}
            for i in INDICES.values() if domain is None or i.domain == domain]
    return pd.DataFrame(rows)


def index(obj: xr.Dataset | xr.DataArray, name: str | Iterable[str], *,
          sensor: str | None = None, bands: Mapping[str, str] | None = None,
          **params: float) -> xr.DataArray | xr.Dataset:
    """Compute one or more spectral indices.

    Parameters
    ----------
    obj : Dataset with one variable per band, or DataArray with a ``band`` dim.
        Bands should be scaled reflectance (see :func:`geolattice.rsense.optical.scale`).
    name : index name (case-insensitive) or a list of names.
    sensor : ``landsat`` / ``landsat89``, ``landsat_tm``, ``sentinel2``, ``modis``,
        ``planetscope``, ``planetscope8``, ``worldview2``, ``worldview3``, ``naip``,
        ``hls_l30`` or ``hls_s30``; guessed if omitted.
    bands : explicit canonical-to-variable mapping, e.g. ``{"nir": "B8A"}``.
    params : override formula constants, e.g. ``L=0.25`` for SAVI.

    Returns a DataArray (one index) or Dataset (several). Data stay on the
    active backend, so on CUDA the result is CuPy-backed; use ``rs.to_cpu``.
    """
    if not isinstance(name, str):
        return xr.Dataset({n: index(obj, n, sensor=sensor, bands=bands, **params) for n in name})
    key = ALIASES.get(name.upper(), name.upper())
    if key not in INDICES:
        raise KeyError(f"Unknown index {name!r}. See geolattice.rsense.optical.list_indices().")
    spec = INDICES[key]
    on_gpu = get_backend().array_gpu
    arrays = {}
    for b in spec.bands:
        da = get_band(obj, b, sensor=sensor, bands=bands)
        if not np.issubdtype(da.dtype, np.floating):
            da = da.astype("float32")
        arrays[b] = to_device(da) if on_gpu else da
    kw = {**spec.params, **{k: v for k, v in params.items() if k in spec.params}}
    unknown = set(params) - set(spec.params)
    if unknown:
        raise TypeError(f"{spec.name} has no parameter(s) {sorted(unknown)}; "
                        f"valid: {sorted(spec.params)}")
    with np.errstate(divide="ignore", invalid="ignore"):
        out = spec.formula(**arrays, **kw)
    out = out.where(np.isfinite(out))
    out.name = spec.name
    out.attrs = {"long_name": spec.long_name, "reference": spec.reference,
                 "pyrsense:index": spec.name}
    return out
