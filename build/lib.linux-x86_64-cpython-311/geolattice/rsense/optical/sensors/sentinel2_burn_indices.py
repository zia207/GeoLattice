"""Sentinel-2 burn indices for forest fire detection and severity assessment.

This module provides functions to calculate burn-related spectral indices from
Sentinel-2 imagery, including NBR, dNBR, NBR2, BAI, BAIS2, and MIRBI.

Reference:
- CAL FIRE 2026 Incident Archive and Sentinel-2 burn severity mapping workflows
- Key and Benson (2006) for NBR/dNBR
- García-Martín et al. (2018) for BAIS2
- Hardtke et al. (2019) for BAI and MIRBI
"""

from __future__ import annotations

import numpy as np
import xarray as xr


def nbr(b8: xr.DataArray, b12: xr.DataArray) -> xr.DataArray:
    """Normalized Burn Ratio (NBR).

    NBR compares vegetation-sensitive NIR reflectance with SWIR2 reflectance.
    Fire damage often decreases NBR because vegetation structure and moisture change.

    Formula: NBR = (B8 - B12) / (B8 + B12)

    Parameters
    ----------
    b8 : xr.DataArray
        Near-infrared (NIR) band (842 nm, 10 m resolution)
    b12 : xr.DataArray
        Shortwave infrared 2 (SWIR2) band (2190 nm, 20 m resolution)

    Returns
    -------
    xr.DataArray
        NBR values ranging approximately from -1 to 1.

    Reference
    ---------
    Key, C.H. and Benson, N.C. (2006). Landscape assessment: ground measure
    of severity, the Composite Burn Index; and remote sensing of severity,
    the Normalized Burn Ratio. In: Fires Effects Guide. USDA Forest Service,
    Rocky Mountain Research Station.
    """
    return (b8 - b12) / (b8 + b12)


def dnbr(nbr_pre: xr.DataArray, nbr_post: xr.DataArray) -> xr.DataArray:
    """Differenced Normalized Burn Ratio (dNBR).

    dNBR estimates the magnitude of vegetation change between pre-fire and
    post-fire images. Positive values generally indicate vegetation loss or burn damage.

    Formula: dNBR = NBR_pre - NBR_post

    Parameters
    ----------
    nbr_pre : xr.DataArray
        Pre-fire NBR values
    nbr_post : xr.DataArray
        Post-fire NBR values

    Returns
    -------
    xr.DataArray
        dNBR values. Higher positive values often correspond to stronger vegetation
        and surface change. Negative values may indicate greening or other changes.

    Reference
    ---------
    Key, C.H. and Benson, N.C. (2006). Landscape assessment: ground measure
    of severity, the Composite Burn Index; and remote sensing of severity,
    the Normalized Burn Ratio. In: Fires Effects Guide. USDA Forest Service,
    Rocky Mountain Research Station.
    """
    return nbr_pre - nbr_post


def nbr2(b11: xr.DataArray, b12: xr.DataArray) -> xr.DataArray:
    """Normalized Burn Ratio 2 (NBR2).

    NBR2 examines changes in surface and vegetation moisture using two SWIR bands.
    It can help distinguish burned surfaces from unburned vegetation and assess
    changes associated with drying.

    Formula: NBR2 = (B11 - B12) / (B11 + B12)

    Parameters
    ----------
    b11 : xr.DataArray
        Shortwave infrared 1 (SWIR1) band (1610 nm, 20 m resolution)
    b12 : xr.DataArray
        Shortwave infrared 2 (SWIR2) band (2190 nm, 20 m resolution)

    Returns
    -------
    xr.DataArray
        NBR2 values ranging approximately from -1 to 1.

    Notes
    -----
    NBR2 is particularly appropriate for 20 m analysis since both SWIR bands
    are natively 20 m resolution. Soil type and background moisture can affect
    its performance.
    """
    return (b11 - b12) / (b11 + b12)


def bai(b4: xr.DataArray, b8: xr.DataArray, ref_red: float = 0.1,
        ref_nir: float = 0.06) -> xr.DataArray:
    """Burned Area Index (BAI).

    BAI highlights pixels whose red and NIR reflectance resembles the spectral
    characteristics of burned surfaces. It emphasizes pixels near reference
    reflectance values associated with burned surfaces.

    Formula: BAI = 1 / ((ref_red - B4)^2 + (ref_nir - B8)^2)

    Parameters
    ----------
    b4 : xr.DataArray
        Red band (665 nm, 10 m resolution)
    b8 : xr.DataArray
        Near-infrared (NIR) band (842 nm, 10 m resolution)
    ref_red : float, optional
        Reference red reflectance for burned surfaces, default 0.1
    ref_nir : float, optional
        Reference NIR reflectance for burned surfaces, default 0.06

    Returns
    -------
    xr.DataArray
        BAI values. Higher values can indicate potential burned areas.

    Notes
    -----
    BAI is sensitive to reflectance scaling. The default reference values 0.1
    and 0.06 assume reflectance is expressed as a fraction (0.10 rather than
    1000 in scaled Sentinel-2 data). Adjust these values if using different
    reflectance scales.

    Reference
    ---------
    Hardtke, L.A., et al. (2019). Burned area index for mapping forest fire
    severity in eastern Amazonia.
    """
    return 1.0 / ((ref_red - b4) ** 2 + (ref_nir - b8) ** 2)


def bais2(b4: xr.DataArray, b6: xr.DataArray, b7: xr.DataArray,
          b8a: xr.DataArray, b12: xr.DataArray) -> xr.DataArray:
    """Burned Area Index for Sentinel-2 (BAIS2).

    BAIS2 uses visible, red-edge, NIR, and SWIR bands to improve burned-area
    detection. It was developed specifically for Sentinel-2 imagery and can
    provide additional discrimination between burned and unburned surfaces.

    Formula: BAIS2 = (1 - sqrt(B6 * B7 * B8A / B4)) * ((B12 - B8A) / sqrt(B12 + B8A) + 1)

    Parameters
    ----------
    b4 : xr.DataArray
        Red band (665 nm, 10 m resolution)
    b6 : xr.DataArray
        Red-edge 1 band (740 nm, 20 m resolution)
    b7 : xr.DataArray
        Red-edge 2 band (783 nm, 20 m resolution)
    b8a : xr.DataArray
        Narrow NIR band (865 nm, 20 m resolution)
    b12 : xr.DataArray
        Shortwave infrared 2 (SWIR2) band (2190 nm, 20 m resolution)

    Returns
    -------
    xr.DataArray
        BAIS2 values. Higher values indicate higher likelihood of burned areas.

    Reference
    ---------
    García-Martín, A., et al. (2018). Burned area mapping algorithm for
    Sentinel-2 data (BAIS2). In: 2018 IEEE International Geoscience and
    Remote Sensing Symposium (IGARSS).
    """
    term1 = 1.0 - np.sqrt((b6 * b7 * b8a) / b4)
    term2 = (b12 - b8a) / np.sqrt(b12 + b8a) + 1.0
    return term1 * term2


def mirbi(b12: xr.DataArray, b11: xr.DataArray) -> xr.DataArray:
    """Mid-Infrared Burn Index (MIRBI).

    MIRBI separates burned surfaces from unburned backgrounds using SWIR
    reflectance. It uses the contrast between the two SWIR bands and is
    especially useful in dry environments where burned ground, ash, and dry
    vegetation have distinct SWIR responses.

    Formula: MIRBI = 10 * B12 - 9.8 * B11 + 2

    Parameters
    ----------
    b12 : xr.DataArray
        Shortwave infrared 2 (SWIR2) band (2190 nm, 20 m resolution)
    b11 : xr.DataArray
        Shortwave infrared 1 (SWIR1) band (1610 nm, 20 m resolution)

    Returns
    -------
    xr.DataArray
        MIRBI values. Higher values indicate stronger burn signatures.

    Notes
    -----
    MIRBI is not a normalized index and its numerical range depends on the
    input reflectance scale. Use the reflectance scaling and formulation
    appropriate to the source of your imagery.

    Reference
    ---------
    Hardtke, L.A., et al. (2019). Burned area index for mapping forest fire
    severity in eastern Amazonia.
    """
    return 10.0 * b12 - 9.8 * b11 + 2.0


__all__ = [
    "nbr",
    "dnbr",
    "nbr2",
    "bai",
    "bais2",
    "mirbi",
]
