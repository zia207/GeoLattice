"""Hyperspectral image processing (imaging spectroscopy).

Cubes are ``xarray.DataArray`` objects with dims ``(band, y, x)`` and a ``wavelength`` (nm)
band coordinate (plus ``fwhm`` and ``good`` when known); see :mod:`.cube`.

Modules: ``cube`` (helpers), ``io`` (ENVI, EMIT, EnMAP, PRISMA, Tanager, OHS, AVIRIS,
GeoTIFF, swath gridding, Earthdata search), ``preprocess`` (bad bands, noise, destriping,
masks, empirical line, smoothing, derivatives, continuum removal, band depth, spectral
resampling), ``reduce`` (PCA, MNF, ICA, HySime, virtual dimensionality, band selection),
``indices`` (narrowband indices by wavelength, red-edge position), ``library`` (USGS /
ECOSTRESS / CSV libraries), ``match`` (SAM, SID, SCA), ``endmembers`` (PPI, ATGP, N-FINDR,
VCA), ``unmixing`` (LS, SCLS, NNLS, FCLS, MESMA, sparse), ``detect`` (RX, matched filter,
ACE, CEM, OSP, gas plumes and IME), ``traits`` (PLSR, regressors, GPR, PROSAIL hybrid
retrieval) and ``viz``. Deep learning models live in :mod:`geolattice.rsense.dl.hyperspectral`.
"""

from . import (
    cube,
    detect,
    endmembers,
    indices,
    io,
    library,
    match,
    preprocess,
    reduce,
    traits,
    unmixing,
    viz,
)
from .cube import (
    as_cube,
    band_at,
    band_index,
    band_stats,
    mean_spectrum,
    sel_range,
    spectra_at,
    wavelengths,
)
from .detect import (
    ace,
    cem,
    detection_roc,
    emission_rate,
    gas_matched_filter,
    matched_filter,
    osp,
    plume_ime,
    plume_mask,
    rx,
)
from .endmembers import atgp, extract_endmembers, nfindr, ppi, vca
from .indices import available as available_indices
from .indices import index, list_indices, red_edge_position, register_index
from .io import (
    glt_orthorectify,
    read_aviris,
    read_emit,
    read_enmap,
    read_envi,
    read_envi_header,
    read_geotiff_cube,
    read_hyperspectral,
    read_ohs,
    read_prisma,
    read_tanager,
    read_tanager_ancillary,
    search,
    subset_file,
    swath_to_grid,
    write_envi,
)
from .library import (
    SpectralLibrary,
    from_image,
    merge_libraries,
    read_ecostress,
    read_usgs,
    synthetic_library,
)
from .match import classify_spectral, sam, sid, spectral_similarity
from .preprocess import (
    bad_bands,
    band_depth,
    continuum_removal,
    derivative,
    destripe,
    drop_bad_bands,
    empirical_line,
    estimate_noise,
    flat_field,
    iar_reflectance,
    normalize,
    quality_mask,
    resample_spectral,
    resample_to_sensor,
    smooth,
    snv,
)
from .reduce import hysime, ica, mnf, mnf_denoise, pca, select_bands, virtual_dimensionality
from .traits import (
    TraitModel,
    fit_plsr,
    fit_regressor,
    hybrid_retrieval,
    prosail_lut,
    prosail_spectrum,
    vip_scores,
)
from .unmixing import mesma, reconstruct, solve_abundances, sparse_unmix, unmix
from .viz import plot_band_stats, plot_rgb, plot_spectra, rgb, spectral_viewer, wavelength_slider

__all__ = [
    "SpectralLibrary", "TraitModel", "ace", "as_cube", "atgp", "available_indices", "bad_bands",
    "band_at", "band_depth", "band_index", "band_stats", "cem", "classify_spectral",
    "continuum_removal", "cube", "derivative", "destripe", "detect", "detection_roc",
    "drop_bad_bands", "emission_rate", "empirical_line", "endmembers", "estimate_noise",
    "extract_endmembers", "fit_plsr", "fit_regressor", "flat_field", "from_image",
    "gas_matched_filter", "glt_orthorectify", "hybrid_retrieval", "hysime", "iar_reflectance",
    "ica", "index", "indices", "io", "library", "list_indices", "match", "matched_filter",
    "mean_spectrum", "merge_libraries", "mesma", "mnf", "mnf_denoise", "nfindr", "normalize",
    "osp", "pca", "plot_band_stats", "plot_rgb", "plot_spectra", "plume_ime", "plume_mask",
    "ppi", "preprocess", "prosail_lut", "prosail_spectrum", "quality_mask", "read_aviris",
    "read_ecostress", "read_emit", "read_enmap", "read_envi", "read_envi_header",
    "read_geotiff_cube", "read_hyperspectral", "read_ohs", "read_prisma", "read_tanager",
    "read_tanager_ancillary", "read_usgs", "reconstruct", "red_edge_position", "reduce",
    "register_index", "resample_spectral", "resample_to_sensor", "rgb", "rx", "sam", "search",
    "select_bands", "sel_range", "sid", "smooth", "snv", "solve_abundances", "sparse_unmix",
    "spectra_at", "spectral_similarity", "spectral_viewer", "subset_file", "swath_to_grid",
    "synthetic_library", "traits", "unmix", "unmixing", "vca", "vip_scores",
    "virtual_dimensionality", "viz", "wavelength_slider", "wavelengths", "write_envi",
]
