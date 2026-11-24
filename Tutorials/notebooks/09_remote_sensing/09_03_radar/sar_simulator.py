"""Physically based Sentinel-1 simulator for the Part III notebooks.

Real Sentinel-1 time series and interferometric pairs are large and need online accounts, so
the radar notebooks simulate them **from real landscape data**: land cover is derived from a
real Landsat composite and terrain from a real DEM (central Sierra Nevada, California). The
simulator then applies the physics taught in the modules:

* class- and polarization-specific backscatter with incidence-angle dependence,
* side-looking geometry: local incidence angle, foreshortening/layover and radar shadow,
* multiplicative speckle (Gamma-distributed intensity, or circular-Gaussian complex SLC),
* interferometric phase from deformation, topography (perpendicular baseline) and atmosphere,
  with coherence controlled by land cover and time.

Everything returned is a georeferenced ``xarray`` object, so ``geolattice.rsense`` functions run on
it exactly as on real Sentinel-1 RTC or SLC data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from rasterio.enums import Resampling

import geolattice.rsense as rs

WAVELENGTH = 0.05546576  # Sentinel-1 C-band (m)

# Land-cover classes and reference backscatter at 35° incidence (dB): (VV, VH, angular exponent n)
CLASSES = {1: "Water", 2: "Forest", 3: "Shrub/grass", 4: "Bare/rock", 5: "Snow/ice"}
SIGMA_REF = {1: (-21.0, -29.0, 6.0), 2: (-7.5, -13.0, 1.0), 3: (-11.0, -18.0, 2.0),
             4: (-12.5, -22.0, 2.5), 5: (-14.0, -22.0, 2.0)}
COHERENCE = {1: 0.05, 2: 0.25, 3: 0.55, 4: 0.85, 5: 0.15}       # 12-day C-band coherence
COHERENCE_DECAY_DAYS = {1: 1.0, 2: 15.0, 3: 40.0, 4: 300.0, 5: 10.0}


def study_area(landsat_path, dem_path, bounds=(-119.30, 37.45, -118.60, 38.15), res=90.0):
    """Land cover and DEM on a UTM 11N grid around Mono Lake and Long Valley caldera."""
    img = rs.read(landsat_path, band_names=["blue", "green", "red", "nir"]).astype("float32")
    img = img.where(img > 0) / 255.0
    from pyproj import Transformer

    t = Transformer.from_crs(4326, 32611, always_xy=True)
    x0, y0 = t.transform(bounds[0], bounds[1])
    x1, y1 = t.transform(bounds[2], bounds[3])
    img = img.rio.reproject("EPSG:32611", resolution=res, resampling=Resampling.bilinear)
    img = img.rio.clip_box(x0, y0, x1, y1)
    img = img.where(img > 0)
    dem = rs.read(dem_path).squeeze("band", drop=True).astype("float32")
    dem = dem.rio.reproject_match(img.isel(band=0), resampling=Resampling.bilinear)
    dem = dem.where(dem > -100).rename("elevation")
    b = img.to_dataset(dim="band")
    ndvi = (b.nir - b.red) / (b.nir + b.red)
    bright = (b.blue + b.green + b.red) / 3
    lc = xr.full_like(ndvi, 4, dtype="int8")
    lc = xr.where(ndvi > 0.25, 3, lc)
    lc = xr.where(ndvi > 0.45, 2, lc)
    lc = xr.where((b.nir < 0.12) & (ndvi < 0.05), 1, lc)
    lc = xr.where((bright > 0.75) & (dem > 3000), 5, lc)
    lc = lc.where(dem.notnull(), 0).astype("int8").rename("landcover")
    return xr.Dataset({"landcover": lc, "elevation": dem}).rio.write_crs(32611), b


def incidence_grid(template: xr.DataArray, near=30.0, far=46.0, look="east"):
    """Ellipsoid incidence angle increasing across the swath (looking east = ascending)."""
    x = np.linspace(near, far, template.sizes["x"])
    if look == "west":
        x = x[::-1]
    return xr.DataArray(np.broadcast_to(x, template.shape).copy(), dims=template.dims,
                        coords=template.coords, name="incidence")


def local_geometry(dem: xr.DataArray, incidence: xr.DataArray, heading=-10.0):
    """Local incidence angle (deg) plus layover and shadow masks for a right-looking SAR."""
    slope = np.radians(rs.lidar.slope(dem).values)
    aspect = np.radians(np.nan_to_num(rs.lidar.aspect(dem).values))
    theta = np.radians(incidence.values)
    look_az = np.radians(heading + 90.0)                  # direction the radar beam travels
    to_sensor = look_az + np.pi
    cos_loc = np.cos(theta) * np.cos(slope) + np.sin(theta) * np.sin(slope) * np.cos(aspect - to_sensor)
    loc = np.degrees(np.arccos(np.clip(cos_loc, -1, 1)))
    facing = np.cos(aspect - to_sensor)                   # >0: slope faces the sensor
    layover = (facing > 0) & (slope * facing > theta)
    shadow = loc >= 89.0
    tpl = dem.copy()
    return (tpl.copy(data=loc.astype("float32")).rename("local_incidence"),
            tpl.copy(data=layover).rename("layover"), tpl.copy(data=shadow).rename("shadow"))


def backscatter(landcover: xr.DataArray, local_inc: xr.DataArray, pol="VV", terrain=True,
                incidence: xr.DataArray | None = None, overrides: dict | None = None):
    """Speckle-free sigma0 (linear) for each pixel from class, polarization and geometry."""
    k = 0 if pol.upper() == "VV" else 1
    lc = landcover.values
    ref_db = np.full(lc.shape, np.nan, "float32")
    n = np.ones(lc.shape, "float32")
    table = {**SIGMA_REF, **(overrides or {})}
    for c, vals in table.items():
        ref_db[lc == c] = vals[k]
        n[lc == c] = vals[2]
    theta = np.radians(local_inc.values if terrain else incidence.values)
    s0 = 10 ** (ref_db / 10) * (np.clip(np.cos(theta), 0.02, 1) / np.cos(np.radians(35))) ** n
    if terrain and incidence is not None:   # projected-area brightening of slopes facing the sensor
        s0 = s0 * np.clip(np.sin(np.radians(incidence.values)) / np.clip(np.sin(theta), 0.05, 1), 0.2, 8)
    return landcover.copy(data=s0.astype("float32")).rename(f"sigma0_{pol.lower()}")


def speckle(intensity: xr.DataArray, looks=4.4, seed=0) -> xr.DataArray:
    """Multiplicative Gamma speckle with ``looks`` equivalent looks."""
    rng = np.random.default_rng(seed)
    n = rng.gamma(looks, 1.0 / looks, intensity.shape).astype("float32")
    return intensity * n


def time_series(scene: xr.Dataset, dates, *, incidence, local_inc, events=(), looks=4.4,
                seed=0, wet_dates=()):
    """Dual-pol RTC-like gamma0 time series (linear) with optional land-cover events.

    ``events``: list of dicts {"mask": bool DataArray, "start": date, "end": date|None,
    "class": new class or None, "vv_db": offset, "vh_db": offset}. ``wet_dates``: dates with
    wet snow (strong backscatter drop on snow).
    """
    rng = np.random.default_rng(seed)
    out = {"vv": [], "vh": []}
    for k, d in enumerate(pd.to_datetime(dates)):
        lc = scene.landcover.copy()
        off = {"vv": np.zeros(lc.shape, "float32"), "vh": np.zeros(lc.shape, "float32")}
        for ev in events:
            active = d >= pd.Timestamp(ev["start"]) and (ev.get("end") is None or d < pd.Timestamp(ev["end"]))
            if not active:
                continue
            m = ev["mask"].values
            if ev.get("class") is not None:
                lc = lc.where(~ev["mask"], ev["class"])
            off["vv"][m] += ev.get("vv_db", 0.0)
            off["vh"][m] += ev.get("vh_db", 0.0)
        for pol in ("vv", "vh"):
            s0 = backscatter(lc, local_inc, pol, incidence=incidence).values
            seasonal = 0.6 * np.sin(2 * np.pi * (d.dayofyear - 120) / 365) * (scene.landcover.values == 3)
            wet = -6.0 * (scene.landcover.values == 5) if d in pd.to_datetime(list(wet_dates)) else 0
            s0 = s0 * 10 ** ((off[pol] + seasonal + wet + rng.normal(0, 0.3)) / 10)
            g0 = s0 / np.cos(np.radians(incidence.values))
            g0 = g0 * rng.gamma(looks, 1.0 / looks, g0.shape)
            out[pol].append(g0.astype("float32"))
    coords = {"time": pd.to_datetime(dates), "y": scene.y, "x": scene.x}
    ds = xr.Dataset({p: (("time", "y", "x"), np.stack(v)) for p, v in out.items()}, coords=coords)
    ds = ds.where(scene.landcover > 0).rio.write_crs(32611)
    ds.attrs.update({"pyrsense:sensor": "sentinel1", "units": "linear gamma0", "simulated": "yes"})
    return ds


# ----------------------------------------------------------------------------- InSAR
def fractal_screen(shape, beta=8 / 3, sigma=1.0, seed=0):
    """Isotropic power-law random field (turbulent tropospheric delay), unit std × sigma."""
    rng = np.random.default_rng(seed)
    ky = np.fft.fftfreq(shape[0])[:, None]
    kx = np.fft.fftfreq(shape[1])[None, :]
    k = np.hypot(kx, ky)
    k[0, 0] = np.inf
    amp = k ** (-beta / 2)
    f = np.fft.ifft2(amp * (rng.normal(size=shape) + 1j * rng.normal(size=shape))).real
    return sigma * (f - f.mean()) / f.std()


def mogi(x, y, x0, y0, depth, dv, nu=0.25):
    """Surface displacement (east, north, up; metres) of a Mogi point source of volume change dv."""
    dx, dy = x - x0, y - y0
    r3 = (dx**2 + dy**2 + depth**2) ** 1.5
    c = (1 - nu) * dv / np.pi
    return c * dx / r3, c * dy / r3, c * depth / r3


def los_vector(incidence_deg, heading_deg=-10.0):
    """East, north, up components of the ground-to-satellite unit vector (right-looking)."""
    inc = np.radians(incidence_deg)
    look = np.radians(heading_deg + 90.0)
    return -np.sin(inc) * np.sin(look), -np.sin(inc) * np.cos(look), np.cos(inc)


def slc_pair(intensity: np.ndarray, phase: np.ndarray, coherence: np.ndarray, seed=0):
    """Two correlated complex SLCs: s1·conj(s2) has phase ``phase`` and coherence ``coherence``."""
    rng = np.random.default_rng(seed)
    z1 = (rng.normal(size=phase.shape) + 1j * rng.normal(size=phase.shape)) / np.sqrt(2)
    z2 = (rng.normal(size=phase.shape) + 1j * rng.normal(size=phase.shape)) / np.sqrt(2)
    g = np.clip(coherence, 0, 0.999)
    a = np.sqrt(intensity)
    s1 = a * z1
    s2 = a * (g * z1 + np.sqrt(1 - g**2) * z2) * np.exp(-1j * phase)
    return s1.astype("complex64"), s2.astype("complex64")


def coherence_map(landcover: np.ndarray, dt_days: float, layover=None, shadow=None):
    """Expected coherence from land cover and temporal baseline (exponential decay)."""
    g = np.zeros(landcover.shape, "float32")
    for c, g12 in COHERENCE.items():
        tau = COHERENCE_DECAY_DAYS[c]
        g0 = min(1.0, g12 / np.exp(-12 / tau)) if tau > 1 else g12
        g[landcover == c] = g0 * np.exp(-dt_days / tau) if tau > 1 else g12
    if layover is not None:
        g[layover] = 0.05
    if shadow is not None:
        g[shadow] = 0.0
    return np.clip(g, 0, 0.98)
