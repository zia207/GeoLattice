"""Registry and cache for the pretrained models used by the UAV workflows.

Weights are downloaded only when asked (:func:`fetch`) into ``~/.cache/geolattice/uav`` (or
``$GEOLATTICE_UAV_WEIGHTS``). Google Drive folders (the LUMS RiceNet and YOLOv5 releases) are
fetched with ``gdown``. Check each model's licence before redistribution.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WeightSpec:
    name: str
    url: str
    filename: str
    task: str
    source: str
    kind: str = "file"  # "file" (direct URL) or "gdrive_folder"
    sha256: str | None = None  # full hex digest, checked after download when set


REGISTRY: dict[str, WeightSpec] = {
    "sam_vit_b": WeightSpec(
        "sam_vit_b",
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "sam_vit_b_01ec64.pth",
        "Segment Anything ViT-B (plant masks)",
        "Kirillov et al. 2023, Apache-2.0",
    ),
    "sam_vit_l": WeightSpec(
        "sam_vit_l",
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "sam_vit_l_0b3195.pth",
        "Segment Anything ViT-L",
        "Kirillov et al. 2023, Apache-2.0",
    ),
    "sam_vit_h": WeightSpec(
        "sam_vit_h",
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "sam_vit_h_4b8939.pth",
        "Segment Anything ViT-H",
        "Kirillov et al. 2023, Apache-2.0",
    ),
    "ricenet_count": WeightSpec(
        "ricenet_count",
        "https://drive.google.com/drive/folders/1PBroNSvRrshGoxPPIrnB-WLnW1wAHXn8",
        "checkpoint_counting*",
        "RiceNet rice plant counting (density map)",
        "Chaudhry, Rana & Jaleel 2026 (LUMS), MIT",
        kind="gdrive_folder",
    ),
    "ricenet_size": WeightSpec(
        "ricenet_size",
        "https://drive.google.com/drive/folders/1PBroNSvRrshGoxPPIrnB-WLnW1wAHXn8",
        "checkpoint_size_estimation*",
        "RiceNet plant size regression",
        "Chaudhry, Rana & Jaleel 2026 (LUMS), MIT",
        kind="gdrive_folder",
    ),
    "yolov5_rice_panicle": WeightSpec(
        "yolov5_rice_panicle",
        "https://drive.google.com/drive/folders/1SRSeG2DGEHrifDp7ZYn038eRH3VBjvbI",
        "*.pt",
        "YOLOv5 rice panicle detection (use backend='yolov5')",
        "Chaudhry, Rana & Jaleel 2026 (LUMS), MIT",
        kind="gdrive_folder",
    ),
}


def cache_dir() -> Path:
    d = Path(os.environ.get("GEOLATTICE_UAV_WEIGHTS", Path.home() / ".cache" / "geolattice" / "uav"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def list_weights():
    """Table of registered models and whether each is already cached."""
    import pandas as pd

    rows = []
    for s in REGISTRY.values():
        rows.append(
            {
                "name": s.name,
                "task": s.task,
                "source": s.source,
                "kind": s.kind,
                "cached": _find(s) is not None,
            }
        )
    return pd.DataFrame(rows)


def _find(spec: WeightSpec) -> Path | None:
    d = cache_dir() / spec.name
    if not d.exists():
        return None
    hits = sorted(d.rglob(spec.filename))
    return hits[0] if hits else None


def _check(path: Path, digest: str | None) -> None:
    if not digest:
        return
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    if h.hexdigest() != digest.lower():
        raise OSError(f"{path.name}: sha256 mismatch")


def fetch(name: str, *, force: bool = False, verify: bool = True) -> Path:
    """Path to the cached weights for ``name``, downloading them first if needed.

    ``name`` is a key of :data:`REGISTRY` (see :func:`list_weights`).
    """
    if name not in REGISTRY:
        raise KeyError(f"unknown weights {name!r}; choose from {sorted(REGISTRY)}")
    spec = REGISTRY[name]
    found = None if force else _find(spec)
    if found is not None:
        return found
    target = cache_dir() / spec.name
    target.mkdir(parents=True, exist_ok=True)
    if spec.kind == "gdrive_folder":
        try:
            import gdown
        except ImportError as e:
            raise ImportError(f"{name} is on Google Drive; `pip install gdown` first") from e
        gdown.download_folder(url=spec.url, output=str(target), quiet=False, use_cookies=False)
    else:
        import urllib.request

        dest = target / spec.filename
        tmp = dest.with_suffix(dest.suffix + ".part")
        urllib.request.urlretrieve(spec.url, tmp)  # noqa: S310  fixed https URLs from the registry
        tmp.replace(dest)
    found = _find(spec)
    if found is None:
        raise FileNotFoundError(f"downloaded {spec.url} but no {spec.filename} found in {target}")
    if verify:
        _check(found, spec.sha256)
    return found
