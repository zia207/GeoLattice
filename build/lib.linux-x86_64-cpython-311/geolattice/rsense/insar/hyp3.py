"""ASF HyP3 on-demand processing (RTC, InSAR, burst InSAR). Needs ``hyp3_sdk`` and Earthdata login."""

from __future__ import annotations

from os import PathLike
from typing import Any


def _client(username: str | None = None, password: str | None = None):
    try:
        import hyp3_sdk
    except ImportError as e:
        raise ImportError("HyP3 needs `pip install hyp3_sdk` (geolattice[rsense-sar])") from e
    return hyp3_sdk.HyP3(username=username, password=password)


def submit_rtc(granule: str, *, name: str | None = None, resolution: int = 30,
               radiometry: str = "gamma0", scale: str = "power", dem_matching: bool = False,
               speckle_filter: bool = False, include_inc_map: bool = True, **kwargs: Any):
    """Submit a radiometric terrain correction job; returns the ``hyp3_sdk.Batch``."""
    hyp3 = _client(kwargs.pop("username", None), kwargs.pop("password", None))
    return hyp3.submit_rtc_job(granule, name=name, resolution=resolution, radiometry=radiometry,
                               scale=scale, dem_matching=dem_matching,
                               speckle_filter=speckle_filter, include_inc_map=include_inc_map,
                               **kwargs)


def submit_insar(reference: str, secondary: str, *, name: str | None = None, looks: str = "20x4",
                 include_displacement_maps: bool = True, include_inc_map: bool = True,
                 apply_water_mask: bool = True, **kwargs: Any):
    """Submit a GAMMA InSAR job for a granule pair; returns the ``hyp3_sdk.Batch``."""
    hyp3 = _client(kwargs.pop("username", None), kwargs.pop("password", None))
    return hyp3.submit_insar_job(reference, secondary, name=name, looks=looks,
                                 include_displacement_maps=include_displacement_maps,
                                 include_inc_map=include_inc_map,
                                 apply_water_mask=apply_water_mask, **kwargs)


def submit_insar_isce_burst(reference: str, secondary: str, *, name: str | None = None,
                            **kwargs: Any):
    """Submit an ISCE2 burst InSAR job (burst granule ids)."""
    hyp3 = _client(kwargs.pop("username", None), kwargs.pop("password", None))
    return hyp3.submit_insar_isce_burst_job(reference, secondary, name=name, **kwargs)


def submit_sbas_stack(pairs: list[tuple[str, str]], *, name: str, **kwargs: Any):
    """Submit one InSAR job per (reference, secondary) pair for an SBAS network."""
    hyp3 = _client(kwargs.pop("username", None), kwargs.pop("password", None))
    import hyp3_sdk

    batch = hyp3_sdk.Batch()
    for ref, sec in pairs:
        batch += hyp3.submit_insar_job(ref, sec, name=name, **kwargs)
    return batch


def wait_and_download(batch, out_dir: str | PathLike = "hyp3", *, extract: bool = True):
    """Block until jobs finish, then download (and unzip) the products."""
    hyp3 = _client()
    batch = hyp3.watch(batch)
    files = batch.download_files(str(out_dir))
    if extract:
        import zipfile

        for f in files:
            if str(f).endswith(".zip"):
                with zipfile.ZipFile(f) as z:
                    z.extractall(out_dir)
    return files
