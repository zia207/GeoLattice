"""Segment Anything (SAM) segmentation of UAV/satellite imagery via ``segment-geospatial``."""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Any, Literal


def sam_segment(image_path: str | PathLike, out_path: str | PathLike = "sam_masks.tif", *,
                mode: Literal["auto", "boxes", "points", "text"] = "auto", model: str = "vit_h",
                boxes: list | None = None, points: list | None = None, point_labels: list | None = None,
                text_prompt: str | None = None, box_threshold: float = 0.24,
                text_threshold: float = 0.24, vector_path: str | PathLike | None = None,
                device: str | None = None, **kwargs: Any) -> dict:
    """Segment a GeoTIFF with SAM; writes a mask raster and optionally polygons.

    ``auto`` generates all masks; ``boxes``/``points`` prompt with map coordinates
    (image CRS); ``text`` uses Grounding DINO + SAM (LangSAM), e.g. ``text_prompt="tree"``.
    Requires ``pip install segment-geospatial``. GPU strongly recommended. ``image_path`` may
    also be a georeferenced DataArray (written to a temporary GeoTIFF). For PNG/JPG frames
    without georeferencing use :func:`geolattice.rsense.uav.sam_plant_mask` instead.
    """
    import tempfile

    import xarray as xr

    if isinstance(image_path, xr.DataArray):
        if image_path.rio.crs is None:
            raise TypeError("sam_segment needs a georeferenced image; for plain frames use "
                            "rs.uav.sam_plant_mask(image, checkpoint)")
        tmp = Path(tempfile.mkdtemp()) / "sam_input.tif"
        image_path.rio.to_raster(tmp)
        image_path = tmp
    if mode == "points" and not points:
        raise ValueError("mode='points' needs points=[[x, y], ...] in map coordinates")
    if mode == "boxes" and not boxes:
        raise ValueError("mode='boxes' needs boxes=[[xmin, ymin, xmax, ymax], ...]")
    if mode == "text" and not text_prompt:
        raise ValueError("mode='text' needs text_prompt=")
    try:
        import samgeo
    except ImportError as e:
        raise ImportError("sam_segment needs `pip install segment-geospatial`") from e
    from ..backend import get_backend

    # SAM's float64 ops are not supported on Apple MPS, so default to CPU there
    device = device or ("cuda" if get_backend().name == "cuda" else "cpu")
    out_path = str(out_path)
    if mode == "text":
        from samgeo.text_sam import LangSAM

        sam = LangSAM()
        sam.predict(str(image_path), text_prompt, box_threshold=box_threshold,
                    text_threshold=text_threshold, output=out_path, **kwargs)
    elif mode == "auto":
        sam = samgeo.SamGeo(model_type=model, automatic=True, device=device)
        sam.generate(str(image_path), out_path, **kwargs)
    else:
        sam = samgeo.SamGeo(model_type=model, automatic=False, device=device)
        sam.set_image(str(image_path))
        if mode == "boxes":
            sam.predict(boxes=boxes, point_crs=kwargs.pop("crs", None), output=out_path, **kwargs)
        else:
            sam.predict(point_coords=points, point_labels=point_labels or [1] * len(points),
                        point_crs=kwargs.pop("crs", None), output=out_path, **kwargs)
    result = {"mask": Path(out_path)}
    if vector_path:
        samgeo.common.raster_to_vector(out_path, str(vector_path))
        result["vector"] = Path(vector_path)
    return result
