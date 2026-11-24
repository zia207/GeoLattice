"""OpenDroneMap bridge: run photogrammetry via a NodeODM server (pyodm) or Docker."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_OPTIONS = {"dsm": True, "dtm": True, "orthophoto-resolution": 2.0,
                   "dem-resolution": 5.0, "feature-quality": "high", "pc-quality": "medium"}


@dataclass(frozen=True)
class Camera:
    """Nadir camera geometry for flight planning (sensor size in mm, image size in pixels)."""

    name: str
    sensor_width_mm: float
    sensor_height_mm: float
    focal_mm: float
    image_width_px: int
    image_height_px: int
    note: str = ""


def camera_from_equivalent(name: str, focal_35mm: float, image_width_px: int, image_height_px: int,
                           *, note: str = "") -> Camera:
    """Build a :class:`Camera` from a 35 mm-equivalent focal length (diagonal-based), for
    drones that publish only that. The sensor is scaled to a 43.27 mm diagonal, so only
    the ratio sensor/focal (which sets GSD and footprint) is meaningful."""
    diag = 43.27
    r = image_width_px / image_height_px
    h = float(diag / np.hypot(r, 1.0))
    return Camera(name, h * r, h, focal_35mm, image_width_px, image_height_px,
                  note or f"from {focal_35mm:g} mm equivalent; verify against your camera")


CAMERAS: dict[str, Camera] = {
    "phantom4pro": Camera("DJI Phantom 4 Pro", 13.2, 8.8, 8.8, 5472, 3648),
    "mavic3m_rgb": Camera("DJI Mavic 3M RGB", 17.3, 13.0, 12.29, 5280, 3956),
    "p4multispectral": Camera("DJI P4 Multispectral (each band)", 4.87, 3.96, 5.74, 1600, 1300),
    "rededge_mx": Camera("MicaSense RedEdge-MX", 4.8, 3.6, 5.4, 1280, 960),
    "mini4pro": camera_from_equivalent("DJI Mini 4 Pro (48 MP photo)", 24, 8064, 6048),
    "mini4pro_4k_video": camera_from_equivalent("DJI Mini 4 Pro (4K video frame)", 24, 3840, 2160),
}


def flight_plan(altitude_m: float, *, camera: str | Camera | None = None,
                sensor_width_mm: float = 13.2, focal_mm: float = 8.8,
                image_width_px: int = 5472, image_height_px: int = 3648, front_overlap: float = 0.8,
                side_overlap: float = 0.7, speed_ms: float | None = None, fps: float | None = None,
                frame_step: int = 1) -> dict[str, float]:
    """Ground sampling distance, image footprint and line/trigger spacing for a nadir survey.

    ``camera`` picks a preset from :data:`CAMERAS` (or a :class:`Camera`); otherwise the
    sensor arguments are used (defaults: 1" 20 MP, DJI Phantom 4 Pro). The image width is
    assumed across-track. For video capture give ``speed_ms``, ``fps`` and ``frame_step``
    (keep every n-th frame): the plan then adds the along-track ``frame_spacing_m`` and the
    resulting ``video_front_overlap``.
    """
    if camera is not None:
        cam = CAMERAS[camera] if isinstance(camera, str) else camera
        sensor_width_mm, focal_mm = cam.sensor_width_mm, cam.focal_mm
        image_width_px, image_height_px = cam.image_width_px, cam.image_height_px
    gsd = sensor_width_mm * altitude_m / (focal_mm * image_width_px)  # m/px
    w, h = gsd * image_width_px, gsd * image_height_px
    plan = {"gsd_cm": gsd * 100, "footprint_w_m": w, "footprint_h_m": h,
            "line_spacing_m": w * (1 - side_overlap), "trigger_distance_m": h * (1 - front_overlap)}
    if speed_ms:
        plan["trigger_interval_s"] = plan["trigger_distance_m"] / speed_ms
        if fps:
            spacing = speed_ms * frame_step / fps
            plan["frame_spacing_m"] = spacing
            plan["video_front_overlap"] = max(0.0, 1 - spacing / h)
    return plan


def run_nodeodm(images: Sequence[str | PathLike], out_dir: str | PathLike = "odm_output", *,
                host: str = "localhost", port: int = 3000, options: dict[str, Any] | None = None,
                token: str = "") -> Path:
    """Process images on a NodeODM server (``docker run -p 3000:3000 opendronemap/nodeodm``)."""
    try:
        from pyodm import Node
    except ImportError as e:
        raise ImportError("run_nodeodm needs `pip install pyodm` (geolattice[rsense-uav])") from e
    node = Node(host, port, token)
    task = node.create_task([str(p) for p in images], {**DEFAULT_OPTIONS, **(options or {})})
    task.wait_for_completion()
    out = Path(out_dir)
    task.download_assets(str(out))
    return out


def run_docker(project_dir: str | PathLike, *, options: dict[str, Any] | None = None,
               gpu: bool = False, image: str | None = None) -> Path:
    """Run ODM in Docker on ``project_dir/images``; outputs land in ``project_dir``."""
    if shutil.which("docker") is None:
        raise RuntimeError("Docker not found")
    proj = Path(project_dir).resolve()
    img = image or ("opendronemap/odm:gpu" if gpu else "opendronemap/odm")
    cmd = ["docker", "run", "--rm", "-v", f"{proj.parent}:/datasets"]
    if gpu:
        cmd += ["--gpus", "all"]
    cmd += [img, "--project-path", "/datasets", proj.name]
    for k, v in {**DEFAULT_OPTIONS, **(options or {})}.items():
        cmd += [f"--{k}"] if v is True else ([] if v is False else [f"--{k}", str(v)])
    subprocess.run(cmd, check=True)
    return proj


def load_outputs(odm_dir: str | PathLike | None = None, *, orthophoto: str | PathLike | None = None,
                 dsm: str | PathLike | None = None, dtm: str | PathLike | None = None,
                 point_cloud: str | PathLike | None = None, chunks: Any = "auto") -> dict[str, Any]:
    """Load the orthomosaic, DSM and DTM of a photogrammetry project (whichever exist).

    With ``odm_dir`` the OpenDroneMap layout is searched. Explicit paths (e.g. Agisoft
    Metashape or Pix4D exports) override or replace it.
    """
    from ..core.raster import read

    files: dict[str, Path] = {}
    if odm_dir is not None:
        d = Path(odm_dir)
        files = {"orthophoto": d / "odm_orthophoto" / "odm_orthophoto.tif",
                 "dsm": d / "odm_dem" / "dsm.tif", "dtm": d / "odm_dem" / "dtm.tif"}
        pc = d / "odm_georeferencing" / "odm_georeferenced_model.laz"
        if point_cloud is None and pc.exists():
            point_cloud = pc
    for k, v in {"orthophoto": orthophoto, "dsm": dsm, "dtm": dtm}.items():
        if v is not None:
            files[k] = Path(v)
    out: dict[str, Any] = {k: read(v, chunks=chunks) for k, v in files.items() if v.exists()}
    if point_cloud is not None and Path(point_cloud).exists():
        out["point_cloud_path"] = Path(point_cloud)
    return out
