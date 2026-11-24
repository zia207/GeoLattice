"""Hyperspectral I/O: ENVI, EMIT (NetCDF + GLT orthorectification), EnMAP L2A, PRISMA HE5,
Planet Tanager HDF-EOS5 (ortho and basic swath), Orbita OHS band GeoTIFFs (with RPC
orthorectification), AVIRIS-NG/AVIRIS-3 (ENVI), band-described GeoTIFFs, swath gridding from
per-pixel lat/lon, and NASA Earthdata search.

All readers return a cube (see :mod:`.cube`): ``(band, y, x)`` with ``wavelength`` (nm),
``fwhm`` and ``good`` band coordinates, fill values as NaN and a CRS when known.
"""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from .cube import as_cube

# ----------------------------------------------------------------------------- ENVI
ENVI_DTYPES = {1: "u1", 2: "i2", 3: "i4", 4: "f4", 5: "f8", 12: "u2", 13: "u4", 14: "i8",
               15: "u8"}
_DTYPE_ENVI = {np.dtype(v).str[1:]: k for k, v in ENVI_DTYPES.items()}


def read_envi_header(path: str | PathLike) -> dict[str, Any]:
    """Parse an ENVI ``.hdr`` into a dict (lists for ``{...}`` values, numbers converted)."""
    p = Path(path)
    hdr = p if p.suffix.lower() == ".hdr" else _find_hdr(p)
    text = hdr.read_text(errors="ignore")
    if not text.lstrip().upper().startswith("ENVI"):
        raise ValueError(f"{hdr} is not an ENVI header")
    out: dict[str, Any] = {}
    for m in re.finditer(r"^\s*([^=\n]+?)\s*=\s*(\{.*?\}|[^\n]*)", text, flags=re.M | re.S):
        key, val = m.group(1).strip().lower(), m.group(2).strip()
        if val.startswith("{"):
            items = [s.strip() for s in val[1:-1].replace("\n", " ").split(",")]
            out[key] = [_num(s) for s in items if s != ""]
        else:
            out[key] = _num(val)
    return out


def _num(s: str):
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s


def _find_hdr(p: Path) -> Path:
    for c in (p.with_suffix(".hdr"), Path(str(p) + ".hdr")):
        if c.exists():
            return c
    raise FileNotFoundError(f"No ENVI header next to {p}")


def _find_data(hdr: Path) -> Path:
    stem = hdr.with_suffix("")
    for c in (stem, *[stem.with_suffix(e) for e in (".img", ".dat", ".bsq", ".bil", ".bip",
                                                     ".raw", ".IMG", ".DAT")]):
        if c.exists() and c != hdr:
            return c
    raise FileNotFoundError(f"No ENVI data file for {hdr}")


def _envi_geo(h: dict) -> tuple[Any, Any]:
    """(affine transform, crs) from ``map info`` / ``coordinate system string``."""
    from rasterio.transform import Affine

    mi = h.get("map info")
    if not mi or len(mi) < 7:
        return None, None
    proj = str(mi[0]).lower()
    refx, refy, east, north, px, py = (float(v) for v in mi[1:7])
    rot = 0.0
    for item in mi[7:]:
        if isinstance(item, str) and item.lower().startswith("rotation"):
            rot = float(item.split("=")[1])
    x0 = east - (refx - 1) * px
    y0 = north + (refy - 1) * py
    T = Affine(px, 0, x0, 0, -py, y0)
    if rot:
        T = T * Affine.rotation(-rot)
    crs = None
    css = h.get("coordinate system string")
    if css:
        crs = css if isinstance(css, str) else ",".join(map(str, css))
    elif "utm" in proj and len(mi) >= 9:
        zone = int(mi[7])
        south = "south" in str(mi[8]).lower()
        crs = f"EPSG:{(32700 if south else 32600) + zone}"
    elif "geographic" in proj:
        crs = "EPSG:4326"
    return T, crs


def read_envi(path: str | PathLike, *, bands: Sequence[int] | None = None,
              window: tuple[slice, slice] | None = None, scale: bool = True,
              mask_bad: bool = False) -> xr.DataArray:
    """Read an ENVI image (BSQ/BIL/BIP, any byte order) as a cube.

    ``bands`` (indices) and ``window`` ``(row_slice, col_slice)`` read only a subset through a
    memory map, so large flight lines need not fit in memory. ``scale`` divides by
    ``reflectance scale factor``; ``data ignore value`` becomes NaN; ``bbl`` sets ``good``
    (``mask_bad=True`` drops those bands).
    """
    p = Path(path)
    hdr = p if p.suffix.lower() == ".hdr" else _find_hdr(p)
    h = read_envi_header(hdr)
    data = _find_data(hdr) if p.suffix.lower() == ".hdr" else p
    R, C, B = int(h["lines"]), int(h["samples"]), int(h["bands"])
    dt = np.dtype(ENVI_DTYPES[int(h.get("data type", 4))]).newbyteorder(
        ">" if int(h.get("byte order", 0)) == 1 else "<")
    il = str(h.get("interleave", "bsq")).lower()
    shape = {"bsq": (B, R, C), "bil": (R, B, C), "bip": (R, C, B)}[il]
    mm = np.memmap(data, dtype=dt, mode="r", offset=int(h.get("header offset", 0)), shape=shape)
    rs, cs = window or (slice(None), slice(None))
    bidx = np.arange(B) if bands is None else np.asarray(bands)
    if il == "bsq":
        a = np.asarray(mm[bidx][:, rs, cs])
    elif il == "bil":
        a = np.asarray(mm[rs][:, bidx, :][:, :, cs]).transpose(1, 0, 2)
    else:
        a = np.asarray(mm[rs, cs][:, :, bidx]).transpose(2, 0, 1)
    a = a.astype("float32")
    ign = h.get("data ignore value")
    if ign is not None:
        a[a == float(ign)] = np.nan
    a[a <= -9990] = np.nan
    sf = h.get("reflectance scale factor")
    if scale and sf not in (None, 0, 1, 1.0):
        a /= float(sf)
    wl = np.asarray(h.get("wavelength", np.arange(B)), float)[bidx]
    units = str(h.get("wavelength units", "nm")).lower()
    um = units.startswith("micro") or (units in ("unknown", "") and np.nanmax(wl) < 30)
    fw = np.asarray(h["fwhm"], float)[bidx] if "fwhm" in h else None
    T, crs = _envi_geo(h)
    r0 = (rs.start or 0) if isinstance(rs, slice) else 0
    c0 = (cs.start or 0) if isinstance(cs, slice) else 0
    coords = {}
    if T is not None and T.b == 0 and T.d == 0:
        coords["y"] = T.f + T.e * (r0 + np.arange(a.shape[1]) + 0.5)
        coords["x"] = T.c + T.a * (c0 + np.arange(a.shape[2]) + 0.5)
    da = xr.DataArray(a, dims=("band", "y", "x"), coords=coords, name="reflectance",
                      attrs={k: v for k, v in h.items() if isinstance(v, (int, float, str))
                             and k not in ("description",)})
    cube = as_cube(da, wl, fwhm=fw, units="um" if um else "nm")
    good = np.asarray(h["bbl"], float)[bidx] > 0 if "bbl" in h else np.ones(len(bidx), bool)
    cube = cube.assign_coords(good=("band", good))
    if crs is not None:
        cube = cube.rio.write_crs(crs)
    if mask_bad:
        cube = cube.isel(band=np.flatnonzero(good))
    return cube


def write_envi(cube: xr.DataArray, path: str | PathLike, *, interleave: str = "bsq",
               dtype: str = "float32", description: str = "geolattice.rsense.hyperspectral") -> str:
    """Write a cube as ENVI (``path`` + ``.hdr``); keeps wavelengths, FWHM, bad bands, map
    info and the CRS (WKT). Returns the data file path."""
    p = Path(path)
    data = p.with_suffix("") if p.suffix.lower() == ".hdr" else p
    a = np.asarray(cube.values, dtype=dtype)
    B, R, C = a.shape
    order = {"bsq": (0, 1, 2), "bil": (1, 0, 2), "bip": (1, 2, 0)}[interleave]
    np.ascontiguousarray(a.transpose(order)).astype(np.dtype(dtype).newbyteorder("<")).tofile(data)
    lines = ["ENVI", f"description = {{{description}}}", f"samples = {C}", f"lines = {R}",
             f"bands = {B}", "header offset = 0", "file type = ENVI Standard",
             f"data type = {_DTYPE_ENVI[np.dtype(dtype).str[1:]]}", f"interleave = {interleave}",
             "byte order = 0", "data ignore value = -9999"]
    if "wavelength" in cube.coords:
        lines += ["wavelength units = Nanometers",
                  "wavelength = {" + ", ".join(f"{w:.4f}" for w in cube["wavelength"].values) + "}"]
    if "fwhm" in cube.coords:
        lines.append("fwhm = {" + ", ".join(f"{w:.4f}" for w in cube["fwhm"].values) + "}")
    if "good" in cube.coords:
        lines.append("bbl = {" + ", ".join(str(int(g)) for g in cube["good"].values) + "}")
    if cube.rio.crs is not None and "x" in cube.dims:
        T = cube.rio.transform()
        lines.append(f"map info = {{Arbitrary, 1, 1, {T.c}, {T.f}, {T.a}, {-T.e}, units=Meters}}")
        lines.append("coordinate system string = {" + cube.rio.crs.to_wkt() + "}")
    with open(str(data) + ".hdr" if data.suffix else data.with_suffix(".hdr"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    # the NaN -> -9999 swap keeps ENVI readers happy
    if np.isnan(a).any():
        mm = np.memmap(data, dtype=np.dtype(dtype).newbyteorder("<"), mode="r+")
        mm[np.isnan(mm)] = -9999
        mm.flush()
    return str(data)


def read_aviris(path: str | PathLike, **kwargs: Any) -> xr.DataArray:
    """AVIRIS-Classic / AVIRIS-NG / AVIRIS-3 reflectance or radiance (ENVI files)."""
    return read_envi(path, **kwargs)


# ----------------------------------------------------------------------------- EMIT
def read_emit(path: str | PathLike, *, ortho: bool = True, variable: str | None = None,
              mask_bad: bool = False, bbox: tuple[float, float, float, float] | None = None
              ) -> xr.DataArray:
    """NASA EMIT L1B radiance / L2A reflectance NetCDF as a cube.

    ``ortho=True`` places the swath on the product's geographic grid (EPSG:4326) with its
    geometry lookup table (GLT). ``bbox`` (lon/lat) crops after orthorectification. The
    ``good_wavelengths`` flag becomes ``good``.
    """
    import h5py

    with h5py.File(path, "r") as f:
        var = variable or next(v for v in ("reflectance", "radiance", "reflectance_uncertainty")
                               if v in f)
        a = np.asarray(f[var][:], dtype="float32")              # (downtrack, crosstrack, bands)
        sbp = f["sensor_band_parameters"]
        wl = np.asarray(sbp["wavelengths"][:], float)
        fw = np.asarray(sbp["fwhm"][:], float) if "fwhm" in sbp else None
        good = np.asarray(sbp["good_wavelengths"][:], bool) if "good_wavelengths" in sbp \
            else np.ones(len(wl), bool)
        loc = f["location"] if "location" in f else None
        glt_x = np.asarray(loc["glt_x"][:]) if loc is not None and "glt_x" in loc else None
        glt_y = np.asarray(loc["glt_y"][:]) if loc is not None and "glt_y" in loc else None
        gt = _attr(f, "geotransform")
        wkt = _attr(f, "spatial_ref")
    a[a <= -9990] = np.nan
    a = np.moveaxis(a, -1, 0)
    if ortho and glt_x is not None and gt is not None:
        valid = (glt_x > 0) & (glt_y > 0)
        out = np.full((a.shape[0], *glt_x.shape), np.nan, dtype="float32")
        out[:, valid] = a[:, glt_y[valid] - 1, glt_x[valid] - 1]
        gt = np.asarray(gt, float)
        H, W = glt_x.shape
        x = gt[0] + gt[1] * (np.arange(W) + 0.5)
        y = gt[3] + gt[5] * (np.arange(H) + 0.5)
        da = xr.DataArray(out, dims=("band", "y", "x"), coords={"y": y, "x": x}, name=var)
        cube = as_cube(da, wl, fwhm=fw).assign_coords(good=("band", good))
        cube = cube.rio.write_crs(wkt.decode() if isinstance(wkt, bytes) else (wkt or "EPSG:4326"))
        if bbox is not None:
            cube = cube.sel(x=slice(bbox[0], bbox[2]), y=slice(bbox[3], bbox[1]))
    else:
        da = xr.DataArray(a, dims=("band", "line", "sample"), name=var)
        cube = as_cube(da, wl, fwhm=fw).assign_coords(good=("band", good))
    return cube.isel(band=np.flatnonzero(good)) if mask_bad else cube


def _attr(f, name):
    v = f.attrs.get(name)
    if v is None and name in f:
        v = f[name][()]
    return v


def glt_orthorectify(swath: xr.DataArray, glt_x: np.ndarray, glt_y: np.ndarray,
                     geotransform: Sequence[float], crs: Any = "EPSG:4326") -> xr.DataArray:
    """Place a ``(band, line, sample)`` swath on a map grid with a geometry lookup table
    (1-based ``glt_x``/``glt_y``, 0 = no data), as used by EMIT and AVIRIS-NG L1B/L2."""
    a = np.asarray(swath.values)
    valid = (glt_x > 0) & (glt_y > 0)
    out = np.full((a.shape[0], *glt_x.shape), np.nan, dtype="float32")
    out[:, valid] = a[:, glt_y[valid] - 1, glt_x[valid] - 1]
    gt = np.asarray(geotransform, float)
    H, W = glt_x.shape
    da = xr.DataArray(out, dims=("band", "y", "x"),
                      coords={"y": gt[3] + gt[5] * (np.arange(H) + 0.5),
                              "x": gt[0] + gt[1] * (np.arange(W) + 0.5),
                              **{c: swath[c] for c in ("band", "wavelength", "fwhm", "good")
                                 if c in swath.coords}}, name=swath.name)
    return da.rio.write_crs(crs)


# ----------------------------------------------------------------------------- EnMAP
def read_enmap(path: str | PathLike, *, metadata: str | PathLike | None = None,
               scale: bool = True, mask_bad: bool = False) -> xr.DataArray:
    """EnMAP L1C/L2A product: a folder (or its ``*SPECTRAL_IMAGE*`` raster) plus
    ``*METADATA.XML`` for band centres, FWHM, gains and offsets. With ``scale`` DNs become
    ``DN * gain + offset`` (reflectance for L2A)."""
    import xml.etree.ElementTree as ET

    import rioxarray

    p = Path(path)
    img = p if p.is_file() else next(iter(sorted(
        [q for q in p.iterdir() if "SPECTRAL_IMAGE" in q.name.upper()
         and q.suffix.upper() in (".TIF", ".TIFF", ".BSQ", ".BIL", ".BIP", ".GEOTIFF")])), None)
    if img is None:
        raise FileNotFoundError(f"No SPECTRAL_IMAGE raster in {p}")
    meta = Path(metadata) if metadata else next(iter(sorted(
        (img.parent).glob("*METADATA*.XML"))), None)
    da = rioxarray.open_rasterio(img, masked=True).astype("float32")
    wl = fw = gain = off = None
    if meta is not None and meta.exists():
        root = ET.parse(meta).getroot()
        bands = root.findall(".//bandCharacterisation/bandID")
        if bands:
            def num(b, tag, default=np.nan):
                e = b.find(tag)
                return float(e.text) if e is not None and e.text else default

            wl = np.array([num(b, "wavelengthCenterOfBand") for b in bands])
            fw = np.array([num(b, "FWHMOfBand") for b in bands])
            gain = np.array([num(b, "GainOfBand", 1.0) for b in bands])
            off = np.array([num(b, "OffsetOfBand", 0.0) for b in bands])
    if wl is None:
        raise ValueError("Band wavelengths not found; pass metadata= (METADATA.XML)")
    crs = da.rio.crs
    if scale and gain is not None:
        da = da * gain[:, None, None] + off[:, None, None]
    cube = as_cube(da.drop_vars("band"), wl, fwhm=fw).rio.write_crs(crs)
    good = np.isfinite(wl) & ~_water_vapour(wl)
    cube = cube.assign_coords(good=("band", good))
    return cube.isel(band=np.flatnonzero(good)) if mask_bad else cube


def _water_vapour(wl: np.ndarray) -> np.ndarray:
    return ((wl > 1340) & (wl < 1455)) | ((wl > 1790) & (wl < 1960)) | (wl > 2450)


# ----------------------------------------------------------------------------- PRISMA
def read_prisma(path: str | PathLike, *, cubes: Sequence[str] = ("VNIR", "SWIR"),
                level: str | None = None, mask_bad: bool = False) -> xr.DataArray:
    """PRISMA L2B/L2C/L2D HE5 reflectance as one VNIR + SWIR cube (sorted by wavelength).

    DNs are scaled with ``L2Scale{Vnir,Swir}{Min,Max}``; zero-wavelength and flagged bands are
    marked ``good=False``. L2D products get a UTM grid from the product corner attributes;
    other levels keep ``(line, sample)`` with ``lat``/``lon`` coordinates.
    """
    import h5py

    with h5py.File(path, "r") as f:
        A = f.attrs
        lvl = level or ("L2D" if any("L2D" in k for k in _walk(f)) else
                        next((s for s in ("L2C", "L2B", "L1") if any(s in k for k in _walk(f))),
                             "L2D"))
        swath = next(k for k in _walk(f) if k.endswith("/Data Fields") and "HCO" in k)
        geo = swath.replace("Data Fields", "Geolocation Fields")
        parts, wls, fws, flags = [], [], [], []
        for c in cubes:
            arr = np.asarray(f[f"{swath}/{c}_Cube"][:], dtype="float32")    # (rows, bands, cols)
            lo, hi = (float(A.get(f"L2Scale{c.title()}Min", 0.0)),
                      float(A.get(f"L2Scale{c.title()}Max", 65535.0)))
            arr = lo + arr * (hi - lo) / 65535.0
            parts.append(arr.transpose(1, 0, 2))
            cw = np.asarray(A[f"List_Cw_{c.title()}"], float)
            wls.append(cw)
            fws.append(np.asarray(A.get(f"List_Fwhm_{c.title()}", np.full(cw.size, np.nan)), float))
            flags.append(np.asarray(A.get(f"List_Cw_{c.title()}_Flags", np.ones(cw.size)), bool))
        lat = np.asarray(f[f"{geo}/Latitude"][:]) if f"{geo}/Latitude" in f else None
        lon = np.asarray(f[f"{geo}/Longitude"][:]) if f"{geo}/Longitude" in f else None
        epsg = A.get("Epsg_Code")
        ulx, uly = A.get("Product_ULcorner_easting"), A.get("Product_ULcorner_northing")
    a = np.concatenate(parts, 0)
    wl, fw, fl = np.concatenate(wls), np.concatenate(fws), np.concatenate(flags)
    order = np.argsort(np.where(wl > 0, wl, np.inf))
    a, wl, fw, fl = a[order], wl[order], fw[order], fl[order]
    good = fl & (wl > 0)
    a[:, np.all(a <= 0, axis=0)] = np.nan          # no-data pixels (all bands zero)
    if lvl == "L2D" and epsg is not None and ulx is not None:
        res = 30.0
        R, C = a.shape[1:]
        da = xr.DataArray(a, dims=("band", "y", "x"),
                          coords={"y": float(uly) - res * (np.arange(R) + 0.5),
                                  "x": float(ulx) + res * (np.arange(C) + 0.5)})
        cube = as_cube(da, np.where(wl > 0, wl, np.nan), fwhm=fw).rio.write_crs(int(epsg))
    else:
        da = xr.DataArray(a, dims=("band", "line", "sample"))
        cube = as_cube(da, np.where(wl > 0, wl, np.nan), fwhm=fw)
        if lat is not None:
            cube = cube.assign_coords(lat=(("line", "sample"), lat), lon=(("line", "sample"), lon))
    cube = cube.assign_coords(good=("band", good)).rename("reflectance")
    return cube.isel(band=np.flatnonzero(good)) if mask_bad else cube


def _walk(f) -> list[str]:
    names: list[str] = []
    f.visit(names.append)
    return names


# ----------------------------------------------------------------------------- Planet Tanager
TANAGER_VARIABLES = ("surface_reflectance", "toa_radiance", "surface_reflectance_uncertainty")


def _tanager_group(f) -> tuple[str, bool]:
    """Data Fields group of a Tanager HDF-EOS5 file and whether it is a grid (ortho)."""
    for kind in ("GRIDS", "SWATHS"):
        g = f"HDFEOS/{kind}/HYP/Data Fields"
        if g in f:
            return g, kind == "GRIDS"
    raise ValueError("Not a Tanager HDF-EOS5 file (no HDFEOS/GRIDS|SWATHS/HYP)")


def _he5_dms(v: float) -> float:
    """HDF-EOS packed DMS (DDDMMMSSS.SS) to decimal degrees."""
    s = np.sign(v) or 1.0
    v = abs(v)
    d, rest = divmod(v, 1e6)
    m, sec = divmod(rest, 1e3)
    return float(s * (d + m / 60 + sec / 3600))


def _he5_grid(struct: str) -> tuple[Any, Any, Any]:
    """CRS and x/y pixel-centre coordinates from an HDF-EOS5 ``StructMetadata.0`` grid."""
    def get(key):
        m = re.search(rf"\b{key}=\(?([^)\n]+)\)?", struct)
        return m.group(1).strip() if m else None

    nx, ny = int(get("XDim")), int(get("YDim"))
    ul = [float(v) for v in get("UpperLeftPointMtrs").split(",")]
    lr = [float(v) for v in get("LowerRightMtrs").split(",")]
    proj = get("Projection") or ""
    if "GEO" in proj:
        ul, lr = [_he5_dms(v) for v in ul], [_he5_dms(v) for v in lr]
        crs = "EPSG:4326"
    elif "UTM" in proj:
        z = int(get("ZoneCode"))
        crs = f"EPSG:{(32700 if z < 0 else 32600) + abs(z)}"
    else:
        raise ValueError(f"Unsupported HDF-EOS5 projection {proj!r}")
    dx, dy = (lr[0] - ul[0]) / nx, (lr[1] - ul[1]) / ny
    return crs, ul[0] + dx * (np.arange(nx) + 0.5), ul[1] + dy * (np.arange(ny) + 0.5)


def _bbox_window(bbox, x: np.ndarray, y: np.ndarray, crs: Any) -> tuple[slice, slice]:
    """Row/column window of a lon/lat ``bbox`` on a projected grid."""
    from pyproj import Transformer

    t = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    lon = np.linspace(bbox[0], bbox[2], 21)
    lat = np.linspace(bbox[1], bbox[3], 21)
    edge_lon = np.r_[lon, lon, np.full(21, bbox[0]), np.full(21, bbox[2])]
    edge_lat = np.r_[np.full(21, bbox[1]), np.full(21, bbox[3]), lat, lat]
    X, Y = t.transform(edge_lon, edge_lat)
    cols = np.flatnonzero((x >= np.min(X)) & (x <= np.max(X)))
    rows = np.flatnonzero((y >= np.min(Y)) & (y <= np.max(Y)))
    if not len(rows) or not len(cols):
        raise ValueError("bbox does not overlap the scene")
    return slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1)


def _band_selection(wl: np.ndarray, bands, wavelength_range) -> np.ndarray:
    idx = np.arange(len(wl)) if bands is None else np.unique(np.asarray(bands, int))
    if wavelength_range is not None:
        lo, hi = wavelength_range
        idx = idx[(wl[idx] >= lo) & (wl[idx] <= hi)]
    if not len(idx):
        raise ValueError("No bands selected")
    return idx


def _h5_read(d, idx: np.ndarray, rows: slice, cols: slice) -> np.ndarray:
    """Read bands ``idx`` of a (band, y, x) dataset; contiguous runs are read as slices."""
    if np.all(np.diff(idx) == 1):
        return np.asarray(d[idx[0]:idx[-1] + 1, rows, cols], dtype="float32")
    return np.asarray(d[idx, rows, cols], dtype="float32")


def read_tanager(path: str | PathLike, *, variable: str | None = None,
                 bands: Sequence[int] | None = None,
                 wavelength_range: tuple[float, float] | None = None,
                 window: tuple[slice, slice] | None = None,
                 bbox: tuple[float, float, float, float] | None = None,
                 mask_bad: bool = False, mask_clouds: bool = False,
                 geolocation: str | PathLike | None = None) -> xr.DataArray:
    """Planet Tanager HDF-EOS5 product (``*_ortho_sr_hdf5.h5``, ``*_ortho_radiance_hdf5.h5``
    or the unorthorectified ``*_basic_radiance_hdf5.h5``) as a cube.

    Wavelengths, FWHM and ``good_wavelengths`` come from the variable's attributes (when a
    product has no good-band flag, the water vapour bands are marked bad). ``variable`` is
    ``surface_reflectance``, ``toa_radiance`` (W m⁻² sr⁻¹ µm⁻¹) or
    ``surface_reflectance_uncertainty``; by default the first one present.

    Ortho products are placed on their UTM grid from ``StructMetadata.0``. Basic (swath)
    products keep ``(band, line, sample)`` with ``lat``/``lon`` coordinates from the file or
    from a ``geolocation`` GeoTIFF (bands: longitude, latitude); see :func:`swath_to_grid`.

    Subset while reading to save memory: ``bands`` (indices), ``wavelength_range`` (nm),
    ``window`` (row and column slices) or, for ortho products, ``bbox`` (lon/lat). The fill
    value and ``nodata_pixels`` become NaN; ``mask_clouds`` also blanks the beta cloud and
    cirrus masks. ``mask_bad`` drops bands flagged bad.
    """
    import h5py

    with h5py.File(path, "r") as f:
        grp, is_grid = _tanager_group(f)
        F = f[grp]
        var = variable or next((v for v in TANAGER_VARIABLES if v in F), None)
        if var is None or var not in F:
            raise KeyError(f"{variable or 'cube variable'} not in {list(F)}")
        d = F[var]
        A = d.attrs
        if "wavelengths" not in A:          # e.g. the uncertainty layer: use the cube's metadata
            A = next(F[v].attrs for v in TANAGER_VARIABLES if v in F and "wavelengths" in F[v].attrs)
        wl = np.asarray(A["wavelengths"], float)
        fw = np.asarray(A["fwhm"], float) if "fwhm" in A else None
        good = np.asarray(A["good_wavelengths"], bool) if "good_wavelengths" in A \
            else ~_water_vapour(wl)
        fill = float(np.asarray(d.attrs.get("_FillValue", A.get("_FillValue", -9999.0))).ravel()[0])
        unit = d.attrs.get("Unit", A.get("Unit", ""))
        unit = unit.decode() if isinstance(unit, bytes) else str(unit)
        crs = x = y = None
        if is_grid:
            crs, x, y = _he5_grid(f["HDFEOS INFORMATION/StructMetadata.0"][()].decode())
            if bbox is not None:
                window = _bbox_window(bbox, x, y, crs)
        rows, cols = window if window is not None else (slice(None), slice(None))
        idx = _band_selection(wl, bands, wavelength_range)
        a = _h5_read(d, idx, rows, cols)
        bad = np.zeros(a.shape[1:], bool)
        for m in ("nodata_pixels",) + (("beta_cloud_mask", "beta_cirrus_mask")
                                       if mask_clouds else ()):
            if m in F:
                bad |= np.asarray(F[m][rows, cols]) == 1
        lat = lon = None
        if not is_grid and geolocation is None:
            gl = grp.replace("Data Fields", "Geolocation Fields")
            if f"{gl}/Latitude" in f:
                lat = np.asarray(f[f"{gl}/Latitude"][rows, cols], float)
                lon = np.asarray(f[f"{gl}/Longitude"][rows, cols], float)
    a[(a == fill) | ~np.isfinite(a)] = np.nan
    a[:, bad] = np.nan
    wl, good = wl[idx], good[idx]
    fw = fw[idx] if fw is not None else None
    if is_grid:
        da = xr.DataArray(a, dims=("band", "y", "x"), coords={"y": y[rows], "x": x[cols]},
                          name=var)
        cube = as_cube(da, wl, fwhm=fw).rio.write_crs(crs)
    else:
        da = xr.DataArray(a, dims=("band", "line", "sample"), name=var)
        cube = as_cube(da, wl, fwhm=fw)
        if geolocation is not None:
            import rasterio

            with rasterio.open(geolocation) as src:
                win = None if window is None else ((rows.start or 0, rows.stop or src.height),
                                                   (cols.start or 0, cols.stop or src.width))
                g = src.read(window=win).astype(float)
                g[g == (src.nodata if src.nodata is not None else -9999.0)] = np.nan
            lon, lat = g[0], g[1]
        if lat is not None:
            lat[lat <= -9990], lon[lon <= -9990] = np.nan, np.nan
            cube = cube.assign_coords(lat=(("line", "sample"), lat),
                                      lon=(("line", "sample"), lon))
    cube = cube.assign_coords(good=("band", good))
    cube.attrs.update(sensor="Tanager", units=unit, product=Path(path).name)
    return cube.isel(band=np.flatnonzero(good)) if mask_bad else cube


def read_tanager_ancillary(path: str | PathLike, *, window: tuple[slice, slice] | None = None,
                           bbox: tuple[float, float, float, float] | None = None
                           ) -> xr.Dataset:
    """All 2-D layers of a Tanager product (cloud, cirrus and no-data masks, aerosol optical
    depth, column water vapour, sun and sensor angles, path length, time) as a Dataset on the
    same grid (or swath) as :func:`read_tanager`."""
    import h5py

    with h5py.File(path, "r") as f:
        grp, is_grid = _tanager_group(f)
        F = f[grp]
        crs = x = y = None
        if is_grid:
            crs, x, y = _he5_grid(f["HDFEOS INFORMATION/StructMetadata.0"][()].decode())
            if bbox is not None:
                window = _bbox_window(bbox, x, y, crs)
        rows, cols = window if window is not None else (slice(None), slice(None))
        dims = ("y", "x") if is_grid else ("line", "sample")
        out = {}
        for k in F:
            if F[k].ndim != 2:
                continue
            v = np.asarray(F[k][rows, cols])
            fv = F[k].attrs.get("_FillValue")
            if v.dtype.kind == "f" and fv is not None:
                v = np.where(v == np.asarray(fv).ravel()[0], np.nan, v)
            out[k] = (dims, v)
    ds = xr.Dataset(out)
    if is_grid:
        ds = ds.assign_coords(y=y[rows], x=x[cols]).rio.write_crs(crs)
    return ds


# ----------------------------------------------------------------------------- swath gridding
def _utm_crs(lon: float, lat: float) -> str:
    z = int((lon + 180) // 6) % 60 + 1
    return f"EPSG:{(32600 if lat >= 0 else 32700) + z}"


def swath_to_grid(swath: xr.DataArray, resolution: float = 30.0, *, crs: Any = "utm",
                  lat: np.ndarray | None = None, lon: np.ndarray | None = None,
                  max_distance: float | None = None) -> xr.DataArray:
    """Nearest-neighbour gridding of a ``(band, line, sample)`` swath with per-pixel
    latitude/longitude (``lat``/``lon`` coordinates or arrays), as in Tanager basic or
    PRISMA L2B/L2C products. ``crs="utm"`` picks the UTM zone of the scene centre;
    ``resolution`` is in ``crs`` units. Grid cells farther than ``max_distance`` (default
    ``resolution * √2``) from any pixel centre stay NaN."""
    from pyproj import Transformer
    from scipy.spatial import cKDTree

    lat = np.asarray(swath["lat"].values if lat is None else lat, float)
    lon = np.asarray(swath["lon"].values if lon is None else lon, float)
    ok = np.isfinite(lat) & np.isfinite(lon)
    if crs == "utm":
        crs = _utm_crs(float(np.nanmean(lon)), float(np.nanmean(lat)))
    X, Y = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(lon[ok], lat[ok])
    x0, x1 = np.floor(X.min() / resolution) * resolution, np.ceil(X.max() / resolution) * resolution
    y0, y1 = np.floor(Y.min() / resolution) * resolution, np.ceil(Y.max() / resolution) * resolution
    gx = np.arange(x0 + resolution / 2, x1, resolution)
    gy = np.arange(y1 - resolution / 2, y0, -resolution)
    GX, GY = np.meshgrid(gx, gy)
    dist, nn = cKDTree(np.c_[X, Y]).query(np.c_[GX.ravel(), GY.ravel()])
    hit = dist <= (max_distance if max_distance is not None else resolution * np.sqrt(2))
    src = np.asarray(swath.values).reshape(swath.shape[0], -1)[:, np.flatnonzero(ok.ravel())]
    out = np.full((swath.shape[0], GX.size), np.nan, dtype="float32")
    out[:, hit] = src[:, nn[hit]]
    da = xr.DataArray(out.reshape(swath.shape[0], *GX.shape), dims=("band", "y", "x"),
                      coords={"y": gy, "x": gx,
                              **{c: swath[c] for c in ("band", "wavelength", "fwhm", "good")
                                 if c in swath.coords}},
                      name=swath.name, attrs=swath.attrs)
    return da.rio.write_crs(crs)


# ----------------------------------------------------------------------------- OHS (Orbita)
# Nominal band centres (nm) of the Zhuhai-1 Orbita hyperspectral satellites (OHS-A to D,
# OHS-2/3), 32 bands at 10 m; pass ``wavelengths=`` to use the calibration file values.
OHS_WAVELENGTHS = (466.0, 480.0, 500.0, 520.0, 536.0, 550.0, 566.0, 580.0, 596.0, 610.0,
                   626.0, 640.0, 656.0, 670.0, 686.0, 700.0, 716.0, 730.0, 746.0, 760.0,
                   776.0, 790.0, 806.0, 820.0, 836.0, 850.0, 866.0, 880.0, 896.0, 910.0,
                   926.0, 940.0)


def _ohs_files(path: Path) -> dict[int, Path]:
    folder = path if path.is_dir() else path.parent
    out = {}
    for q in folder.iterdir():
        m = re.search(r"_B(\d+)(?:_CCD\d+)?\.tiff?$", q.name, re.I)
        if m:
            out[int(m.group(1))] = q
    if not out:
        raise FileNotFoundError(f"No *_B<n>*.tif band files in {folder}")
    return dict(sorted(out.items()))


def _ohs_meta(folder: Path) -> dict[str, Any]:
    import xml.etree.ElementTree as ET

    meta = next(iter(sorted(folder.glob("*meta*.xml"))), None)
    if meta is None:
        return {}
    info = ET.parse(meta).getroot().find("ProductInfo")
    keys = ("SatelliteID", "Sensor", "ProductLevel", "CenterTime", "TimeType", "CloudPercent",
            "ImageRowGSD", "CenterLatitude", "CenterLongitude", "PixelInvalidVal")
    return {k: info.findtext(k) for k in keys if info is not None and info.findtext(k)}


def read_ohs(path: str | PathLike, *, bands: Sequence[int] | None = None,
             window: tuple[slice, slice] | None = None,
             wavelengths: Sequence[float] | None = None, fwhm: Sequence[float] | None = None,
             ortho: bool = False, resolution: float | None = None, crs: Any = "utm",
             height: float | None = None, nodata: float | None = None,
             scale: float | None = None) -> xr.DataArray:
    """Orbita OHS (Zhuhai-1) L1 scene stored as one GeoTIFF per band (``*_B1_*.tif`` to
    ``*_B32_*.tif`` with RPCs) as a cube.

    ``path`` is the scene folder or any band file. ``bands`` are 1-based OHS band numbers;
    ``window`` (row and column slices) reads a part of the image. Wavelengths default to
    :data:`OHS_WAVELENGTHS`. The invalid value from ``*_meta.xml`` (0) becomes NaN and
    ``scale`` multiplies the DNs (e.g. radiometric gain).

    Without ``ortho`` the cube stays in sensor geometry, ``(band, line, sample)``, with the
    RPC model in ``attrs["rpcs"]``. ``ortho=True`` warps it with the RPCs (constant
    ``height``, default the RPC height offset) to ``crs`` (``"utm"`` = zone of the scene
    centre) at ``resolution`` (default the product GSD, 10 m).
    """
    import copy

    import rasterio

    p = Path(path)
    files = _ohs_files(p)
    meta = _ohs_meta(next(iter(files.values())).parent)
    nums = list(files) if bands is None else [int(b) for b in bands]
    missing = [b for b in nums if b not in files]
    if missing:
        raise KeyError(f"OHS bands not found: {missing}")
    if wavelengths is None:
        if max(nums) > len(OHS_WAVELENGTHS):
            raise ValueError("Pass wavelengths= for band numbers above 32")
        wl = np.array([OHS_WAVELENGTHS[b - 1] for b in nums])
    else:
        wl = np.asarray(wavelengths, float)
        if len(wl) != len(nums):                     # one value per OHS band number
            wl = wl[np.asarray(nums) - 1]
    invalid = nodata if nodata is not None else float(meta.get("PixelInvalidVal", 0))
    with rasterio.open(files[nums[0]]) as src:
        H, W = src.height, src.width
        rpcs = src.rpcs
    rows, cols = window if window is not None else (slice(None), slice(None))
    r0, r1, _ = rows.indices(H)
    c0, c1, _ = cols.indices(W)
    win = ((r0, r1), (c0, c1))
    a = np.empty((len(nums), r1 - r0, c1 - c0), dtype="float32")
    for i, b in enumerate(nums):
        with rasterio.open(files[b]) as src:
            a[i] = src.read(1, window=win)
    a[a == invalid] = np.nan
    if scale:
        a *= scale
    if rpcs is not None and (r0 or c0):
        rpcs = copy.copy(rpcs)
        rpcs.line_off -= r0
        rpcs.samp_off -= c0
    attrs = {"sensor": "OHS", **meta}
    if not ortho:
        da = xr.DataArray(a, dims=("band", "line", "sample"),
                          coords={"line": np.arange(r0, r1), "sample": np.arange(c0, c1)},
                          name="radiance", attrs={**attrs, "rpcs": rpcs})
        return as_cube(da, wl, fwhm=fwhm)
    if rpcs is None:
        raise ValueError("ortho=True needs RPCs in the band files")
    from rasterio.warp import Resampling, calculate_default_transform, reproject

    if crs == "utm":
        crs = _utm_crs(rpcs.long_off, rpcs.lat_off)
    res = resolution or float(meta.get("ImageRowGSD", 10))
    h = rpcs.height_off if height is None else height
    T, w, hh = calculate_default_transform("EPSG:4326", crs, a.shape[2], a.shape[1],
                                           rpcs=rpcs, resolution=res, RPC_HEIGHT=h)
    out = np.full((len(nums), hh, w), np.nan, dtype="float32")
    reproject(a, out, rpcs=rpcs, src_crs="EPSG:4326", dst_crs=crs, dst_transform=T,
              src_nodata=np.nan, dst_nodata=np.nan, resampling=Resampling.bilinear,
              RPC_HEIGHT=h)
    x = T.c + T.a * (np.arange(w) + 0.5)
    y = T.f + T.e * (np.arange(hh) + 0.5)
    da = xr.DataArray(out, dims=("band", "y", "x"), coords={"y": y, "x": x}, name="radiance",
                      attrs=attrs)
    return as_cube(da, wl, fwhm=fwhm).rio.write_crs(crs)


# ----------------------------------------------------------------------------- GeoTIFF
def read_geotiff_cube(path: str | PathLike, wavelengths: Sequence[float] | None = None, *,
                      fwhm: Sequence[float] | None = None, scale: float | None = None
                      ) -> xr.DataArray:
    """Multiband GeoTIFF as a cube. Wavelengths come from ``wavelengths=``, the band
    descriptions (e.g. ``"Band 12 (561.3 nm)"``) or band tags ``wavelength``/``WAVELENGTH``."""
    import rasterio
    import rioxarray

    da = rioxarray.open_rasterio(path, masked=True).astype("float32")
    if wavelengths is None:
        with rasterio.open(path) as src:
            wl = []
            for i in range(1, src.count + 1):
                tags = {k.lower(): v for k, v in src.tags(i).items()}
                d = src.descriptions[i - 1] or ""
                wl.append(_parse_wavelength(tags.get("wavelength") or d))
        wavelengths = wl
    if scale:
        da = da * scale
    return as_cube(da.drop_vars("band"), wavelengths, fwhm=fwhm).rio.write_crs(da.rio.crs)


def _parse_wavelength(text: str) -> float:
    """Wavelength in nm from a tag/description: prefers a number with a unit (``561.3 nm``,
    ``0.56 um``), then a number in parentheses, then a bare number (>= 300 means nm)."""
    if not text:
        return np.nan
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(nm|nanometers?|um|µm|micrometers?)\b", text, re.I)
    if m:
        return float(m.group(1)) * (1000.0 if m.group(2).lower()[0] in "uµm" and
                                    not m.group(2).lower().startswith("n") else 1.0)
    m = re.search(r"\(\s*([0-9]+(?:\.[0-9]+)?)\s*\)", text)
    if m:
        return float(m.group(1))
    nums = [float(v) for v in re.findall(r"[0-9]+(?:\.[0-9]+)?", text)]
    big = [v for v in nums if v >= 300]
    return big[-1] if big else (nums[-1] * 1000 if nums and nums[-1] < 3 else np.nan)


def read_hyperspectral(path: str | PathLike, **kwargs: Any) -> xr.DataArray:
    """Open any supported product by inspecting the path: ENVI (``.hdr``/``.img``/no
    extension), EMIT (``EMIT_*.nc``), PRISMA (``.he5``), Tanager (``*_hdf5.h5`` or any
    ``.h5`` with an ``HDFEOS/.../HYP`` group), OHS (folder or band file ``*_B<n>*.tif`` of a
    ``HAZ``/``HBZ``/``HCZ``/``HDZ`` scene), EnMAP (folder or ``ENMAP*``), GeoTIFF."""
    p = Path(path)
    name = p.name.upper()
    if _is_ohs(p):
        return read_ohs(p, **kwargs)
    if p.is_dir() or name.startswith("ENMAP"):
        return read_enmap(p, **kwargs)
    suf = p.suffix.lower()
    if suf in (".nc", ".nc4") or name.startswith("EMIT"):
        return read_emit(p, **kwargs)
    if suf in (".he5", ".h5") and "PRS" in name:
        return read_prisma(p, **kwargs)
    if suf in (".h5", ".he5") and _is_tanager(p):
        return read_tanager(p, **kwargs)
    if suf in (".tif", ".tiff"):
        return read_geotiff_cube(p, **kwargs)
    return read_envi(p, **kwargs)


def _is_tanager(p: Path) -> bool:
    import h5py

    try:
        with h5py.File(p, "r") as f:
            return any(g in f for g in ("HDFEOS/GRIDS/HYP", "HDFEOS/SWATHS/HYP"))
    except OSError:
        return False


def _is_ohs(p: Path) -> bool:
    if not p.exists():
        return False
    pat = re.compile(r"^H[A-D]Z\d?_.*_B\d+(?:_CCD\d+)?\.tiff?$", re.I)
    if p.is_dir():
        return any(pat.match(q.name) for q in p.iterdir())
    return bool(pat.match(p.name))


# ----------------------------------------------------------------------------- search
EARTHDATA_PRODUCTS = {
    "emit_l1b_rad": "EMITL1BRAD", "emit_l2a_rfl": "EMITL2ARFL", "emit_l2a_mask": "EMITL2ARFL",
    "emit_ch4_enh": "EMITL2BCH4ENH", "emit_ch4_plume": "EMITL2BCH4PLM",
    "emit_co2_enh": "EMITL2BCO2ENH", "emit_minerals": "EMITL2BMIN",
    "aviris_ng_l2": "AVIRIS-NG_L2", "pace_oci_l2_aop": "PACE_OCI_L2_AOP",
}


def search(product: str, *, aoi: Any = None, dates: Any = None, **kwargs: Any):
    """Search NASA Earthdata (EMIT, AVIRIS-NG, PACE OCI...) with a short key from
    :data:`EARTHDATA_PRODUCTS` or any CMR short name. Download with
    ``geolattice.rsense.access.earthdata_download``."""
    from ..access import earthdata_search

    return earthdata_search(EARTHDATA_PRODUCTS.get(product.lower(), product), aoi=aoi,
                            dates=dates, **kwargs)


def subset_file(path: str | PathLike, out: str | PathLike, *, window: tuple[slice, slice],
                bands: Sequence[int] | None = None) -> str:
    """Cut a spatial/spectral subset out of a large ENVI file into a new ENVI file (a small,
    reproducible sample cube for tutorials and tests)."""
    cube = read_envi(path, window=window, bands=bands, scale=False)
    os.makedirs(Path(out).parent, exist_ok=True)
    return write_envi(cube, out)
