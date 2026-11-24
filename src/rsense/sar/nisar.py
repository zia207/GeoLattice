"""Readers and analysis helpers for NISAR HDF5 products.

The module supports the eight NISAR products RSLC, RIFG, RUNW, ROFF, GSLC,
GCOV, GUNW, and GOFF. Level-1 products remain in radar coordinates; Level-2
products are read on their geocoded grid. ISCE3, SNAPHU, and MintPy workflows
remain external tools for precision geocoding, phase unwrapping, and time-series
inversion.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

_PRODUCTS = {"RSLC", "RIFG", "RUNW", "ROFF", "GSLC", "GCOV", "GUNW", "GOFF"}
_LEVEL1 = {"RSLC", "RIFG", "RUNW", "ROFF"}


def _h5py():
    """Import the optional HDF5 dependency only when a NISAR file is opened."""
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("NISAR HDF5 reading requires h5py; install `geolattice[rsense-sar]`.") from exc
    return h5py


def _decode(value: Any) -> Any:
    """Convert HDF5 scalar/array metadata to ordinary Python values."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _decode(value.item())
        if value.dtype.kind in "SO":
            return np.asarray([_decode(v) for v in value])
        return value
    if isinstance(value, np.generic):
        return _decode(value.item())
    return value


def _attrs(obj: Any) -> dict[str, Any]:
    return {
        str(key): _decode(value)
        for key, value in obj.attrs.items()
        if key not in {"DIMENSION_LIST", "REFERENCE_LIST"}
    }


def _read_hdf5_dataset(dataset: Any, *, masked: bool = True, scale: bool = False) -> np.ndarray:
    """Read values, handling fill values and optional CF-style scale/offset."""
    values = dataset[()]
    attrs = _attrs(dataset)
    if masked:
        fills = [attrs[key] for key in ("_FillValue", "fill_value", "missing_value") if key in attrs]
        if fills:
            bad = np.zeros(values.shape, dtype=bool)
            for fill in fills:
                try:
                    bad |= values == fill
                except (TypeError, ValueError):
                    continue
            if np.any(bad):
                if values.dtype.kind == "c":
                    values = values.astype(np.result_type(values.dtype, np.complex64), copy=True)
                    values[bad] = np.nan + 1j * np.nan
                elif values.dtype.kind in "iu":
                    values = values.astype("float32")
                    values[bad] = np.nan
                elif values.dtype.kind == "f":
                    values = values.copy()
                    values[bad] = np.nan
    if scale:
        factor = float(attrs.get("scale_factor", 1.0))
        offset = float(attrs.get("add_offset", 0.0))
        if factor != 1.0 or offset != 0.0:
            values = values * factor + offset
    return np.asarray(values)


def _dataset_dims(shape: tuple[int, ...], *, radar: bool) -> tuple[str, ...]:
    if len(shape) == 2:
        return ("azimuth", "range") if radar else ("y", "x")
    if len(shape) == 3:
        return ("band", "azimuth", "range") if radar else ("band", "y", "x")
    return tuple(f"dim_{i}" for i in range(len(shape)))


def inspect_hdf5(path: str | Path, group: str = "/") -> dict[str, dict[str, Any]]:
    """List datasets below an HDF5 group with shape, dtype, and attributes."""
    h5py = _h5py()
    result: dict[str, dict[str, Any]] = {}
    with h5py.File(path, "r") as handle:
        if group not in handle:
            raise KeyError(f"HDF5 group {group!r} does not exist in {path}")

        def visitor(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Dataset):
                group_path = group.rstrip("/")
                dataset_path = f"{group_path}/{name}" if group_path else f"/{name}"
                result[dataset_path] = {
                    "shape": tuple(obj.shape),
                    "dtype": str(obj.dtype),
                    "attrs": _attrs(obj),
                }

        handle[group].visititems(visitor)
    return result


def read_identification(path: str | Path, product: str | None = None) -> dict[str, Any]:
    """Read NISAR identification metadata and attributes without loading raster arrays."""
    h5py = _h5py()
    with h5py.File(path, "r") as handle:
        product_name = product.upper() if product else None
        candidates = [product_name] if product_name else sorted(_PRODUCTS)
        identification = None
        for candidate in candidates:
            if candidate not in _PRODUCTS:
                raise ValueError(f"Unknown NISAR product {candidate!r}; expected one of {sorted(_PRODUCTS)}")
            group_path = f"/science/LSAR/{candidate}/metadata/identification"
            if group_path in handle:
                product_name = candidate
                identification = handle[group_path]
                break
        if identification is None and "/science/LSAR/identification" in handle:
            identification = handle["/science/LSAR/identification"]
            if product_name is None:
                product_name = str(_attrs(identification).get("productType", "")).upper() or None
        if identification is None:
            raise KeyError(f"No NISAR identification group found in {path}")

        result: dict[str, Any] = {"product": product_name}
        result.update(_attrs(identification))

        def read_small(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Dataset) and obj.size <= 1024:
                value = _read_hdf5_dataset(obj, masked=False)
                result[name.rsplit("/", 1)[-1]] = _decode(value)

        identification.visititems(read_small)
        return result


def read_nisar(
    path: str | Path,
    product: str,
    layer_path: str,
    *,
    frequency: str = "A",
    masked: bool = True,
    scale: bool = False,
) -> xr.DataArray:
    """Read a NISAR layer into an ``xarray.DataArray`` with product grid coordinates.

    ``layer_path`` is relative to ``/science/LSAR/{product}/swaths/frequencyA`` for
    Level-1 products and ``.../grids/frequencyA`` for Level-2 products. It may
    include nested product groups such as ``interferogram/HH/wrappedInterferogram``.
    A ``metadata/...`` path reads product metadata layers such as geolocation grids.
    """
    h5py = _h5py()
    product = product.upper()
    if product not in _PRODUCTS:
        raise ValueError(f"Unknown NISAR product {product!r}; expected one of {sorted(_PRODUCTS)}")
    if not frequency or not frequency[0].isalpha():
        raise ValueError("frequency must be a NISAR frequency label such as 'A' or 'B'")
    layer = layer_path.lstrip("/")
    if layer.startswith("metadata/"):
        base = None
        dataset_path = f"/science/LSAR/{product}/{layer}"
    else:
        grid_group = "swaths" if product in _LEVEL1 else "grids"
        base = f"/science/LSAR/{product}/{grid_group}/frequency{frequency.upper()}"
        dataset_path = f"{base}/{layer}"

    with h5py.File(path, "r") as handle:
        if dataset_path not in handle:
            raise KeyError(f"Dataset {dataset_path!r} not found in {path}")
        dataset = handle[dataset_path]
        values = _read_hdf5_dataset(dataset, masked=masked, scale=scale)
        dims = _dataset_dims(values.shape, radar=product in _LEVEL1)
        coords: dict[str, Any] = {}
        crs = None
        if base is not None and len(values.shape) >= 2:
            row_dim, col_dim = dims[-2:]
            coord_names = (
                ("zeroDopplerTime", "slantRange")
                if product in _LEVEL1
                else ("yCoordinates", "xCoordinates")
            )
            parent_path = dataset.parent.name
            grid_paths = [parent_path]
            while parent_path != base and parent_path != "/":
                parent_path = parent_path.rsplit("/", 1)[0] or "/"
                if parent_path not in grid_paths:
                    grid_paths.append(parent_path)
            for coord_name, dim, size in zip(coord_names, (row_dim, col_dim), values.shape[-2:]):
                coord_path = next(
                    (
                        f"{candidate}/{coord_name}"
                        for candidate in grid_paths
                        if f"{candidate}/{coord_name}" in handle
                        and handle[f"{candidate}/{coord_name}"].shape == (size,)
                    ),
                    None,
                )
                if coord_path is not None:
                    coord_values = _read_hdf5_dataset(handle[coord_path], masked=False)
                    coords[dim] = coord_values
                    coords[coord_name] = (dim, coord_values)
            if product not in _LEVEL1:
                projection_path = next(
                    (
                        f"{candidate}/projection"
                        for candidate in grid_paths
                        if f"{candidate}/projection" in handle
                    ),
                    None,
                )
                if projection_path is not None:
                    projection = _read_hdf5_dataset(handle[projection_path], masked=False)
                    crs = int(np.asarray(projection).item())
        attrs = _attrs(dataset)
        attrs.update({"product": product, "frequency": frequency.upper(), "hdf5_path": dataset_path})
        data = xr.DataArray(values, dims=dims, coords=coords, name=layer.rsplit("/", 1)[-1],
                            attrs=attrs)
        if crs is not None:
            data.attrs["crs"] = f"EPSG:{crs}"
            if len(values.shape) == 2:
                try:
                    import rioxarray  # noqa: F401

                    data = data.rio.write_crs(f"EPSG:{crs}")
                except ImportError:
                    pass
                data.attrs["crs"] = f"EPSG:{crs}"
        return data


def read_l2(
    path: str | Path,
    product: str,
    layer_path: str,
    *,
    frequency: str = "A",
    masked: bool = True,
    scale: bool = False,
) -> xr.DataArray:
    """Read a two-dimensional Level-2 raster layer and attach its map CRS."""
    product = product.upper()
    if product in _LEVEL1:
        raise ValueError(f"{product} is a Level-1 radar-geometry product, not Level 2")
    data = read_nisar(path, product, layer_path, frequency=frequency, masked=masked, scale=scale)
    if data.ndim != 2 or data.dims != ("y", "x"):
        raise ValueError(f"Level-2 map layer must be 2-D, received dims {data.dims}")
    if data.attrs.get("crs") is None:
        raise KeyError(f"Missing projection dataset for {product} frequency {frequency}")
    return data


def read_radar(
    path: str | Path,
    product: str,
    layer_path: str,
    *,
    frequency: str = "A",
    masked: bool = True,
    scale: bool = False,
) -> xr.DataArray:
    """Read a two-dimensional Level-1 radar-geometry layer."""
    product = product.upper()
    if product not in _LEVEL1:
        raise ValueError(f"{product} is a Level-2 map-grid product, not Level 1")
    data = read_nisar(path, product, layer_path, frequency=frequency, masked=masked, scale=scale)
    if data.ndim != 2:
        raise ValueError(f"Radar layer must be 2-D, received dims {data.dims}")
    return data


def read_rslc_backscatter(
    path: str | Path,
    polarization: str,
    *,
    frequency: str = "A",
    incidence_angle: xr.DataArray | np.ndarray | None = None,
    looks: tuple[int, int] = (8, 2),
    filter_size: int | None = None,
) -> xr.Dataset:
    """Make multilooked RSLC beta0 and optional sigma0 backscatter quick looks.

    NISAR RSLC complex samples are treated as calibrated beta0. When an incidence-angle
    layer is supplied on the same pixel grid, sigma0 is ``beta0 * sin(theta)``. Use a
    GCOV product or an ISCE3 RTC workflow for terrain-corrected gamma0.
    """
    if any(not isinstance(look, int) or look < 1 for look in looks):
        raise ValueError("looks must contain positive integer (azimuth, range) factors")
    from .sentinel1 import multilook, to_db

    complex_slc = read_radar(path, "RSLC", polarization, frequency=frequency)
    beta0 = (np.abs(complex_slc) ** 2).rename("beta0")
    beta0.attrs.update({"units": "linear beta0", "polarization": polarization.upper()})
    if looks != (1, 1):
        beta0 = multilook(beta0, looks)
    if filter_size is not None:
        if not isinstance(filter_size, int) or filter_size < 1 or filter_size % 2 == 0:
            raise ValueError("filter_size must be a positive odd integer")
        from .speckle import lee

        beta0 = lee(beta0, size=filter_size)
    beta0_db = to_db(beta0).rename("beta0_db")
    beta0_db.attrs.update({"units": "dB", "polarization": polarization.upper()})
    result: dict[str, xr.DataArray] = {"beta0": beta0, "beta0_db": beta0_db}
    if incidence_angle is not None:
        angle = incidence_angle if isinstance(incidence_angle, xr.DataArray) else xr.DataArray(
            np.asarray(incidence_angle), dims=complex_slc.dims, coords=complex_slc.coords
        )
        if angle.shape != complex_slc.shape:
            raise ValueError("incidence_angle must match the source RSLC pixel grid before multilooking")
        if looks != (1, 1):
            angle = multilook(angle, looks)
        sigma0 = (beta0 * np.sin(np.deg2rad(angle))).rename("sigma0")
        sigma0.attrs.update({"units": "linear sigma0", "polarization": polarization.upper()})
        result.update({"sigma0": sigma0, "sigma0_db": to_db(sigma0).rename("sigma0_db")})
    return xr.Dataset(result)


def process_rifg(
    wrapped_interferogram: xr.DataArray | np.ndarray,
    coherence: xr.DataArray | np.ndarray,
    *,
    min_coherence: float = 0.3,
    looks: tuple[int, int] = (1, 1),
) -> xr.Dataset:
    """Mask a RIFG by coherence and optionally multilook the complex interferogram."""
    if not 0 <= min_coherence <= 1:
        raise ValueError("min_coherence must be between 0 and 1")
    if any(not isinstance(look, int) or look < 1 for look in looks):
        raise ValueError("looks must contain positive integer (azimuth, range) factors")
    if isinstance(wrapped_interferogram, xr.DataArray):
        template = wrapped_interferogram
        if isinstance(coherence, xr.DataArray):
            if template.dims != coherence.dims or any(
                (dim in template.coords) != (dim in coherence.coords)
                or (
                    dim in template.coords
                    and not template.coords[dim].equals(coherence.coords[dim])
                )
                for dim in template.dims
            ):
                raise ValueError("interferogram and coherence coordinates must match")
            coh = coherence
        else:
            coh = template.copy(data=np.asarray(coherence))
        if template.shape != coh.shape:
            raise ValueError("interferogram and coherence shapes must match")
        input_valid = np.isfinite(coh) & (coh >= min_coherence) & np.isfinite(template)
        ifg = template.where(input_valid)
        coh = coh.where(input_valid)
        if looks != (1, 1):
            from .sentinel1 import multilook

            coh = multilook(coh, looks)
            ifg = multilook(ifg, looks)
    else:
        ifg_values = np.asarray(wrapped_interferogram)
        coh_values = np.asarray(coherence)
        if ifg_values.shape != coh_values.shape:
            raise ValueError("interferogram and coherence shapes must match")
        input_valid = np.isfinite(coh_values) & (coh_values >= min_coherence) & np.isfinite(ifg_values)
        ifg_values = np.where(input_valid, ifg_values, np.nan + 1j * np.nan)
        coh_values = np.where(input_valid, coh_values, np.nan)
        if looks != (1, 1):
            from .sentinel1 import multilook

            ifg_values = multilook(ifg_values, looks)
            coh_values = multilook(coh_values, looks)
        ifg = xr.DataArray(ifg_values, dims=("y", "x"))
        coh = xr.DataArray(coh_values, dims=("y", "x"))
    valid = np.isfinite(coh) & (coh >= min_coherence) & np.isfinite(ifg)
    masked_ifg = ifg.where(valid)
    return xr.Dataset({
        "interferogram": masked_ifg,
        "wrapped_phase": masked_ifg.copy(data=np.angle(masked_ifg)).rename("wrapped_phase"),
        "coherence": coh.rename("coherence"),
        "valid": valid.rename("valid"),
    }, attrs={"minimum_coherence": float(min_coherence), "looks": tuple(looks)})


def gslc_interferogram(
    reference: xr.DataArray,
    secondary: xr.DataArray,
    *,
    window: tuple[int, int] = (7, 7),
    min_coherence: float = 0.3,
) -> xr.Dataset:
    """Form a GSLC pair interferogram and local coherence on a shared map grid."""
    if reference.shape != secondary.shape or reference.dims != secondary.dims:
        raise ValueError("GSLC arrays must have matching shapes and dimensions")
    if any(
        (dim in reference.coords) != (dim in secondary.coords)
        or (
            dim in reference.coords
            and not reference.coords[dim].equals(secondary.coords[dim])
        )
        for dim in reference.dims
    ):
        raise ValueError("GSLC coordinate grids must match exactly")
    if len(window) != 2 or any(not isinstance(size, int) or size < 1 for size in window):
        raise ValueError("window must be a pair of positive integers")
    from ..insar.interferogram import interferogram

    result = interferogram(reference, secondary, window=window)
    coherence = result["coherence"]
    if min_coherence < 0 or min_coherence > 1:
        raise ValueError("min_coherence must be between 0 and 1")
    valid = np.isfinite(coherence) & (coherence >= min_coherence)
    ifg = result["interferogram"].where(valid)
    return xr.Dataset({
        "interferogram": ifg,
        "wrapped_phase": ifg.copy(data=np.angle(ifg)).rename("wrapped_phase"),
        "coherence": coherence.rename("coherence"),
        "valid": valid.rename("valid"),
    }, attrs={"window": tuple(window), "minimum_coherence": float(min_coherence)})


def process_unwrapped_phase(
    phase: xr.DataArray | np.ndarray,
    *,
    wavelength: float = 0.24,
    coherence: xr.DataArray | np.ndarray | None = None,
    min_coherence: float = 0.3,
    connected_components: xr.DataArray | np.ndarray | None = None,
    reference_mask: xr.DataArray | np.ndarray | None = None,
    ionosphere_phase: xr.DataArray | np.ndarray | None = None,
    incidence_angle: xr.DataArray | np.ndarray | None = None,
    vertical_only: bool = False,
) -> xr.Dataset:
    """Quality-mask, reference, optionally correct ionosphere, and convert phase to LOS displacement.

    The phase-to-LOS convention follows the attached NISAR notes:
    ``LOS displacement = -wavelength * phase / (4 * pi)``.
    """
    if not np.isfinite(wavelength) or wavelength <= 0:
        raise ValueError("wavelength must be positive")
    values = np.asarray(phase.values if isinstance(phase, xr.DataArray) else phase, dtype="float64")
    if values.ndim != 2:
        raise ValueError("phase must be a 2-D image")
    valid = np.isfinite(values)

    def values_of(layer: xr.DataArray | np.ndarray | None, name: str) -> np.ndarray | None:
        if layer is None:
            return None
        array = np.asarray(layer.values if isinstance(layer, xr.DataArray) else layer)
        if array.shape != values.shape:
            raise ValueError(f"{name} must match the phase shape")
        return array

    coh = values_of(coherence, "coherence")
    components = values_of(connected_components, "connected_components")
    iono = values_of(ionosphere_phase, "ionosphere_phase")
    incidence = values_of(incidence_angle, "incidence_angle")
    reference = values_of(reference_mask, "reference_mask")
    if coh is not None:
        if not 0 <= min_coherence <= 1:
            raise ValueError("min_coherence must be between 0 and 1")
        valid &= np.isfinite(coh) & (coh >= min_coherence)
    if components is not None:
        valid &= np.isfinite(components) & (components != 0)
    if vertical_only and incidence is None:
        raise ValueError("incidence_angle is required when vertical_only=True")

    phase_values = values.copy()
    if iono is not None:
        phase_values -= iono
    if reference is None:
        candidates = valid.copy()
        if components is not None and np.any(candidates):
            labels, counts = np.unique(components[candidates], return_counts=True)
            component = labels[np.argmax(counts)]
            candidates &= components == component
        if coh is not None and np.any(candidates):
            candidates &= coh >= np.nanpercentile(coh[candidates], 75)
    else:
        candidates = np.asarray(reference, dtype=bool) & valid
    if not np.any(candidates):
        raise ValueError("No valid stable reference pixels remain after quality masking")
    reference_phase = float(np.nanmedian(phase_values[candidates]))
    referenced_phase = phase_values - reference_phase
    los = -wavelength * referenced_phase / (4 * np.pi)
    output = {"unwrapped_phase": referenced_phase, "los_displacement": los}
    if vertical_only:
        output["vertical_displacement"] = los / np.cos(np.deg2rad(incidence))
    template = phase if isinstance(phase, xr.DataArray) else xr.DataArray(values, dims=("y", "x"))
    dataset = xr.Dataset({
        name: template.copy(data=np.where(valid, array, np.nan)).rename(name)
        for name, array in output.items()
    })
    dataset["valid"] = template.copy(data=valid).rename("valid")
    dataset.attrs.update({
        "wavelength_m": float(wavelength),
        "reference_phase_rad": reference_phase,
        "minimum_coherence": float(min_coherence) if coh is not None else None,
        "ionosphere_corrected": iono is not None,
        "phase_to_los_convention": "negative wavelength * phase / (4*pi)",
    })
    return dataset


def gcov_indices(
    hh: xr.DataArray,
    hv: xr.DataArray,
    vv: xr.DataArray | None = None,
    *,
    min_power: float = 0.0,
) -> xr.Dataset:
    """Calculate GCOV ratios and RVI from linear powers.

    Pass HH and HV for dual-polarization indices, or also VV for quad-pol ratios/RVI.
    """
    if hh.shape != hv.shape or (vv is not None and hh.shape != vv.shape):
        raise ValueError("GCOV layers must have identical shapes")
    if min_power < 0:
        raise ValueError("min_power cannot be negative")
    valid = (hh > min_power) & (hv >= 0)
    cross = hv.where(valid)
    hh_safe = hh.where(valid)
    dual_rvi = 4 * cross / (hh_safe + cross)
    indices = {
        "hv_hh_ratio": (cross / hh_safe).rename("hv_hh_ratio"),
        "dual_pol_rvi": dual_rvi.rename("dual_pol_rvi"),
    }
    if vv is not None:
        quad_valid = valid & (vv > min_power)
        hh_quad, vv_safe = hh.where(quad_valid), vv.where(quad_valid)
        indices["hh_vv_ratio"] = (hh_quad / vv_safe).rename("hh_vv_ratio")
        indices["quad_pol_rvi"] = (
            8 * cross.where(quad_valid) / (hh_quad + vv_safe + 2 * cross.where(quad_valid))
        ).rename("quad_pol_rvi")
    return xr.Dataset(
        indices,
        attrs={
            "input_units": "linear gamma0",
            "min_power": float(min_power),
            "polarization_mode": "dual" if vv is None else "quad",
        },
    )


def process_offsets(
    along_track_offset: xr.DataArray | np.ndarray,
    slant_range_offset: xr.DataArray | np.ndarray,
    *,
    snr: xr.DataArray | np.ndarray | None = None,
    correlation: xr.DataArray | np.ndarray | None = None,
    incidence_angle: xr.DataArray | np.ndarray | None = None,
    heading: xr.DataArray | np.ndarray | float | None = None,
    units: str = "meters",
    azimuth_spacing: float | None = None,
    range_spacing: float | None = None,
    min_snr: float = 5.0,
    min_correlation: float = 0.0,
    stable_mask: xr.DataArray | np.ndarray | None = None,
    despike_size: int = 3,
    interval_days: float | None = None,
) -> xr.Dataset:
    """Quality-filter NISAR ROFF/GOFF offsets and derive ground-range and EN displacement.

    Offset sign conventions depend on product release and look direction. The east/north
    rotation below follows the equations in the supplied notes; verify signs against known
    motion before interpreting geophysical direction.
    """
    azimuth = np.array(along_track_offset.values if isinstance(along_track_offset, xr.DataArray)
                       else along_track_offset, dtype="float64", copy=True)
    slant = np.array(slant_range_offset.values if isinstance(slant_range_offset, xr.DataArray)
                     else slant_range_offset, dtype="float64", copy=True)
    if azimuth.shape != slant.shape or azimuth.ndim != 2:
        raise ValueError("along-track and slant-range offsets must be matching 2-D arrays")
    if units not in {"meters", "pixels"}:
        raise ValueError("units must be 'meters' or 'pixels'")
    if despike_size < 1 or despike_size % 2 == 0:
        raise ValueError("despike_size must be a positive odd integer")
    if interval_days is not None and interval_days <= 0:
        raise ValueError("interval_days must be positive")

    def get_layer(layer: xr.DataArray | np.ndarray | None, name: str) -> np.ndarray | None:
        if layer is None:
            return None
        array = np.asarray(layer.values if isinstance(layer, xr.DataArray) else layer)
        if array.shape != azimuth.shape:
            raise ValueError(f"{name} must match the offset image shape")
        return array

    quality = np.isfinite(azimuth) & np.isfinite(slant)
    snr_values = get_layer(snr, "snr")
    corr_values = get_layer(correlation, "correlation")
    incidence = get_layer(incidence_angle, "incidence_angle")
    stable = get_layer(stable_mask, "stable_mask")
    if snr_values is not None:
        quality &= np.isfinite(snr_values) & (snr_values >= min_snr)
    if corr_values is not None:
        quality &= np.isfinite(corr_values) & (corr_values >= min_correlation)
    if units == "pixels":
        if azimuth_spacing is None or range_spacing is None:
            raise ValueError("pixel offsets require azimuth_spacing and range_spacing in metres")
        if azimuth_spacing <= 0 or range_spacing <= 0:
            raise ValueError("pixel spacings must be positive")
        azimuth *= azimuth_spacing
        slant *= range_spacing
    if incidence is not None:
        quality &= np.isfinite(incidence) & (incidence > 0) & (incidence < 90)
        range_ground = slant / np.sin(np.deg2rad(incidence))
    else:
        range_ground = slant.copy()
    azimuth[~quality] = np.nan
    range_ground[~quality] = np.nan

    if stable is not None:
        stable_pixels = np.asarray(stable, dtype=bool) & quality
        if np.any(stable_pixels):
            azimuth -= np.nanmedian(azimuth[stable_pixels])
            range_ground -= np.nanmedian(range_ground[stable_pixels])

    from scipy.ndimage import median_filter

    for array in (azimuth, range_ground):
        med = median_filter(np.nan_to_num(array, nan=0.0), size=despike_size, mode="nearest")
        residual = np.abs(array - med)
        neighborhood = median_filter(np.nan_to_num(residual, nan=0.0), size=despike_size, mode="nearest")
        array[residual > np.maximum(5 * neighborhood, 1.0)] = np.nan

    output: dict[str, np.ndarray] = {"along_track_offset_m": azimuth}
    output["ground_range_offset_m" if incidence is not None else "slant_range_offset_m"] = range_ground
    if heading is not None:
        heading_values = np.asarray(heading, dtype="float64")
        if heading_values.ndim and heading_values.shape != azimuth.shape:
            raise ValueError("heading must be scalar or match the offset image shape")
        angle = np.deg2rad(heading_values)
        output["east_offset_m"] = azimuth * np.sin(angle) + range_ground * np.cos(angle)
        output["north_offset_m"] = azimuth * np.cos(angle) - range_ground * np.sin(angle)
        if interval_days is not None:
            output["horizontal_velocity_m_per_year"] = (
                np.hypot(output["east_offset_m"], output["north_offset_m"]) / interval_days * 365.25
            )
    template = along_track_offset if isinstance(along_track_offset, xr.DataArray) else xr.DataArray(
        np.asarray(along_track_offset), dims=("y", "x")
    )
    result = xr.Dataset({
        name: template.copy(data=array).rename(name) for name, array in output.items()
    })
    result["valid"] = template.copy(data=quality).rename("valid")
    result.attrs.update({
        "input_offset_units": units,
        "minimum_snr": float(min_snr) if snr_values is not None else None,
        "minimum_correlation": float(min_correlation) if corr_values is not None else None,
        "interval_days": float(interval_days) if interval_days is not None else None,
        "look_direction_sign_check_required": True,
    })
    return result


def geocode_l1(
    data: xr.DataArray | np.ndarray,
    coordinate_x: xr.DataArray | np.ndarray,
    coordinate_y: xr.DataArray | np.ndarray,
    *,
    epsg: int,
    resolution: float,
    height_index: int | None = None,
    stride: int = 1,
) -> xr.DataArray:
    """Quickly interpolate a Level-1 layer to a regular map grid from geolocation arrays.

    This height-slice interpolation is intended for quick looks over relatively flat
    terrain. For quantitative products, use ISCE3 terrain-aware geocoding.
    """
    values = np.asarray(data.values if isinstance(data, xr.DataArray) else data)
    x = np.asarray(coordinate_x.values if isinstance(coordinate_x, xr.DataArray) else coordinate_x)
    y = np.asarray(coordinate_y.values if isinstance(coordinate_y, xr.DataArray) else coordinate_y)
    if values.ndim != 2:
        raise ValueError("data must be a 2-D radar-grid image")
    if x.ndim == 3 and y.ndim == 3:
        if x.shape != y.shape or x.shape[1:] != values.shape:
            raise ValueError("3-D geolocation cubes must share dimensions and match the radar image")
        idx = x.shape[0] // 2 if height_index is None else height_index
        if idx < 0 or idx >= x.shape[0]:
            raise ValueError("height_index is outside the geolocation cube")
        x, y = x[idx], y[idx]
    elif x.shape != values.shape or y.shape != values.shape:
        raise ValueError("coordinate arrays must match data, or be matching (height, azimuth, range) cubes")
    if not isinstance(epsg, int) or epsg <= 0:
        raise ValueError("epsg must be a positive integer")
    if not np.isfinite(resolution) or resolution <= 0:
        raise ValueError("resolution must be positive")
    if not isinstance(stride, int) or stride < 1:
        raise ValueError("stride must be a positive integer")

    from scipy.interpolate import griddata
    from rasterio.transform import from_origin

    sample = (slice(None, None, stride), slice(None, None, stride))
    xs, ys, vals = x[sample], y[sample], values[sample]
    good = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(vals)
    if not np.any(good):
        raise ValueError("No finite geolocation/data samples to interpolate")
    left = np.floor(np.min(xs[good]) / resolution) * resolution
    right = np.ceil(np.max(xs[good]) / resolution) * resolution
    bottom = np.floor(np.min(ys[good]) / resolution) * resolution
    top = np.ceil(np.max(ys[good]) / resolution) * resolution
    width = max(1, int(round((right - left) / resolution)))
    height = max(1, int(round((top - bottom) / resolution)))
    transform = from_origin(left, top, resolution, resolution)
    grid_x = left + (np.arange(width) + 0.5) * resolution
    grid_y = top - (np.arange(height) + 0.5) * resolution
    gx, gy = np.meshgrid(grid_x, grid_y)
    points = np.column_stack((xs[good], ys[good]))
    output = griddata(points, vals[good], (gx, gy), method="linear")
    if isinstance(data, xr.DataArray):
        name = data.name
        attrs = dict(data.attrs)
    else:
        name, attrs = None, {}
    result = xr.DataArray(output, dims=("y", "x"), coords={"y": grid_y, "x": grid_x},
                          name=name, attrs=attrs)
    try:
        import rioxarray  # noqa: F401

        result = result.rio.write_crs(f"EPSG:{epsg}").rio.write_transform(transform)
    except ImportError:
        result.attrs["crs"] = f"EPSG:{epsg}"
        result.attrs["transform"] = tuple(transform)
    return result


__all__ = [
    "geocode_l1",
    "gcov_indices",
    "gslc_interferogram",
    "inspect_hdf5",
    "process_offsets",
    "process_rifg",
    "process_unwrapped_phase",
    "read_identification",
    "read_l2",
    "read_nisar",
    "read_radar",
    "read_rslc_backscatter",
]
