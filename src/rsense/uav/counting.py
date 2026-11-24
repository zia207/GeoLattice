"""Plant counting and plant size from density maps (RiceNet), tiled over large orthomosaics.

RiceNet (Bai et al. 2023) is a VGG16-BN encoder with two decoders: an attention map and a
density map whose integral is the plant count. A second head (``gap=True``) regresses a
typical plant spacing in pixels, used for plant size. The architecture and tiling follow
the LUMS infrastructure-free phenotyping code (Chaudhry, Rana & Jaleel 2026): tiles of
1024 px with 128 px overlap, resized to 320 px for counting and 400 px for size, cosine-
blended density maps, density peaks as plant centres, duplicates merged across tiles.

Requires PyTorch (``geolattice[rsense-dl]``). Weights: see :func:`geolattice.rsense.uav.weights.fetch`.
"""

from __future__ import annotations

from os import PathLike
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
_VGG_URL = "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth"


def _torch():
    try:
        import torch
        from torch import nn
    except ImportError as e:
        raise ImportError("RiceNet needs PyTorch: `pip install geolattice[rsense-dl]`") from e
    return torch, nn


def build_ricenet(gap: bool = False, *, pretrained_backbone: bool = False):
    """Build the RiceNet network: density + attention maps (``gap=False``) or the plant-size
    regressor (``gap=True``). ``pretrained_backbone`` loads ImageNet VGG16-BN weights (only
    needed for training; released checkpoints contain the full network)."""
    torch, nn = _torch()

    class BaseConv(nn.Module):
        def __init__(self, cin, cout, k, s=1, activation=None, use_bn=False):
            super().__init__()
            self.use_bn, self.activation = use_bn, activation
            self.conv = nn.Conv2d(cin, cout, k, s, k // 2)
            self.conv.weight.data.normal_(0, 0.01)
            self.conv.bias.data.zero_()
            self.bn = nn.BatchNorm2d(cout)
            self.bn.weight.data.fill_(1)
            self.bn.bias.data.zero_()

        def forward(self, x):
            x = self.conv(x)
            if self.use_bn:
                x = self.bn(x)
            return self.activation(x) if self.activation else x

    def C(cin, cout, k=3, act=True):
        return BaseConv(cin, cout, k, 1, activation=nn.ReLU() if act else None, use_bn=True)

    class VGG(nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = nn.MaxPool2d(2, 2)
            self.conv1_1, self.conv1_2 = C(3, 64), C(64, 64)
            self.conv2_1, self.conv2_2 = C(64, 128), C(128, 128)
            self.conv3_1, self.conv3_2, self.conv3_3 = C(128, 256), C(256, 256), C(256, 256)
            self.conv4_1, self.conv4_2, self.conv4_3 = C(256, 512), C(512, 512), C(512, 512)
            self.conv5_1, self.conv5_2, self.conv5_3 = C(512, 512), C(512, 512), C(512, 512)

        def forward(self, x):
            x = self.conv1_2(self.conv1_1(x))
            c2 = self.conv2_2(self.conv2_1(self.pool(x)))
            c3 = self.conv3_3(self.conv3_2(self.conv3_1(self.pool(c2))))
            c4 = self.conv4_3(self.conv4_2(self.conv4_1(self.pool(c3))))
            c5 = self.conv5_3(self.conv5_2(self.conv5_1(self.pool(c4))))
            return c2, c3, c4, c5

    class BackEnd(nn.Module):
        def __init__(self):
            super().__init__()
            self.upsample = nn.Upsample(scale_factor=2)
            self.conv1, self.conv2, self.conv3 = C(1024, 512, 1), C(512, 256), C(256, 256)
            self.conv4, self.conv5, self.conv6 = C(512, 256, 1), C(256, 128), C(128, 128)
            self.conv7, self.conv8, self.conv9 = C(256, 64, 1), C(64, 32), C(32, 32)

        def forward(self, c2, c3, c4, c5):
            x = torch.cat([self.upsample(c5), c4], 1)
            x = self.conv3(self.conv2(self.conv1(x)))
            x = torch.cat([self.upsample(x), c3], 1)
            x = self.conv6(self.conv5(self.conv4(x)))
            x = torch.cat([self.upsample(x), c2], 1)
            return self.conv9(self.conv8(self.conv7(x)))

    class Detect(nn.Module):
        def __init__(self):
            super().__init__()
            self.ac = nn.ReLU()
            self.conv1, self.conv2, self.conv3 = C(512, 128), C(128, 128), C(128, 32)
            self.conv4 = C(32, 1)
            self.adappool = nn.AdaptiveMaxPool2d((50, 80))
            self.line1 = nn.Sequential(nn.Linear(4000, 100), nn.Dropout(p=0.01), nn.ReLU())
            self.line2 = nn.Linear(100, 1)
            for m in self.modules():
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        def forward(self, c2, c3, c4, c5):
            x = self.adappool(self.conv4(self.conv3(self.conv2(self.conv1(c5)))))
            return self.line2(self.line1(x.view(x.size(0), -1)))

    class RiceNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.gap = gap
            self.vgg = VGG()
            if gap:
                self.de = Detect()
            else:
                self.amp, self.dmp = BackEnd(), BackEnd()
                self.conv_att = BaseConv(32, 1, 1, 1, activation=nn.Sigmoid(), use_bn=True)
                self.conv_out = BaseConv(32, 1, 1, 1, activation=None, use_bn=False)

        def forward(self, x):
            feats = self.vgg(x)
            if self.gap:
                return self.de(*feats)
            amp = self.conv_att(self.amp(*feats))
            return self.conv_out(amp * self.dmp(*feats)), amp

    net = RiceNet()
    if pretrained_backbone:
        from torch.utils import model_zoo

        sd = model_zoo.load_url(_VGG_URL)
        old = [
            0,
            1,
            3,
            4,
            7,
            8,
            10,
            11,
            14,
            15,
            17,
            18,
            20,
            21,
            24,
            25,
            27,
            28,
            30,
            31,
            34,
            35,
            37,
            38,
            40,
            41,
        ]
        new = [
            "1_1",
            "1_2",
            "2_1",
            "2_2",
            "3_1",
            "3_2",
            "3_3",
            "4_1",
            "4_2",
            "4_3",
            "5_1",
            "5_2",
            "5_3",
        ]
        nd = {}
        for i, n in enumerate(new):
            c, b = old[2 * i], old[2 * i + 1]
            nd[f"conv{n}.conv.weight"] = sd[f"features.{c}.weight"]
            nd[f"conv{n}.conv.bias"] = sd[f"features.{c}.bias"]
            for k in ("weight", "bias", "running_mean", "running_var"):
                nd[f"conv{n}.bn.{k}"] = sd[f"features.{b}.{k}"]
        net.vgg.load_state_dict(nd, strict=False)
    return net


def _device(device: str | None):
    torch, _ = _torch()
    if device:
        return torch.device(device)
    from ..backend import get_backend

    name = get_backend().name
    if name == "cuda" or torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_ricenet(checkpoint: str | PathLike, *, kind: str = "count", device: str | None = None):
    """Load a RiceNet checkpoint (``kind="count"`` density model or ``"size"`` regressor)."""
    torch, _ = _torch()
    dev = _device(device)
    net = build_ricenet(gap=(kind == "size")).to(dev)
    sd = torch.load(str(checkpoint), map_location=dev, weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    net.load_state_dict(sd)
    return net.eval()


def peaks_from_density(density: np.ndarray, threshold: float = 0.001, size: int = 7) -> np.ndarray:
    """Local maxima of a density map above ``threshold``; returns ``(n, 2)`` ``(row, col)``.

    Flat plateaus count once (their centre), as in the original implementation.
    """
    from scipy.ndimage import find_objects, label, maximum_filter

    d = np.asarray(density, np.float32)
    peaks = (d == maximum_filter(d, size=size)) & (d > threshold)
    lab, _ = label(peaks)
    pts = [
        ((sy.start + sy.stop - 1) / 2, (sx.start + sx.stop - 1) / 2) for sy, sx in find_objects(lab)
    ]
    return np.array(pts, dtype=np.float32).reshape(-1, 2)


def _to_tensor(tile: np.ndarray, size: int, dev):
    torch, _ = _torch()
    from PIL import Image

    im = Image.fromarray(tile).resize((size, size), Image.BILINEAR)
    t = torch.from_numpy(np.asarray(im, np.float32) / 255.0).permute(2, 0, 1)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return ((t - mean) / std).unsqueeze(0).to(dev)


def density_count(
    image: Any,
    model: Any,
    *,
    size_model: Any = None,
    tile: int = 1024,
    overlap: int = 128,
    input_size: int = 320,
    size_input: int = 400,
    peak_threshold: float = 0.004,
    border: int = 10,
    padding: int = 10,
    merge_radius: float = 30,
    device: str | None = None,
    progress: bool = False,
) -> tuple[pd.DataFrame, xr.DataArray]:
    """Count plants on a large RGB image with a density model, tile by tile.

    Parameters
    ----------
    image : ``(band, y, x)`` DataArray from :func:`geolattice.rsense.uav.read_frame` or an
        ``(rows, cols, 3)`` uint8/float array.
    model, size_model : RiceNet networks (:func:`load_ricenet`) or checkpoint paths.
        ``size_model`` is optional; without it ``size_px`` comes from plant spacing only.
    tile, overlap : tile edge and overlap in image pixels.
    peak_threshold : minimum density for a peak to count as a plant.
    border, padding : ignore peaks within ``border`` px of a tile edge and ``padding`` px
        of the image edge (edge plants are found by the neighbouring tile).
    merge_radius : peaks closer than this (pixels) in overlapping tiles are one plant.

    Returns
    -------
    points : DataFrame with ``row, col`` (pixels), ``x, y`` (image coordinates, metres when
        the image has a GSD or CRS), ``size_px`` (adaptive box edge) and ``size_m``.
    density : stitched density map as a DataArray on the image grid.
    """
    torch, _ = _torch()
    from scipy.spatial import cKDTree

    from .io import as_image, to_rgb_array
    from .tiling import cosine_weight, merge_points, sliding_windows

    dev = _device(device)
    if isinstance(model, (str, PathLike)):
        model = load_ricenet(model, kind="count", device=str(dev))
    if isinstance(size_model, (str, PathLike)):
        size_model = load_ricenet(size_model, kind="size", device=str(dev))
    da = image if isinstance(image, xr.DataArray) else as_image(np.asarray(image))
    rgb = to_rgb_array(da, uint8=True)
    H, W = rgb.shape[:2]
    heat = np.zeros((H, W), np.float32)
    wsum = np.zeros((H, W), np.float32)
    pts: list[dict] = []
    wins = list(sliding_windows(H, W, tile, overlap))
    if progress:
        from tqdm.auto import tqdm

        wins = tqdm(wins, desc="RiceNet tiles")
    from PIL import Image

    with torch.no_grad():
        for r0, c0, th, tw in wins:
            t = rgb[r0 : r0 + th, c0 : c0 + tw]
            dm = model(_to_tensor(t, input_size, dev))[0].squeeze().float().cpu().numpy()
            sigma = np.nan
            if size_model is not None:
                pre = float(size_model(_to_tensor(t, size_input, dev)).item())
                sigma = pre * ((tw / size_input) + (th / size_input)) / 2.0
            ph, pw = dm.shape
            for r, c in peaks_from_density(dm, peak_threshold):
                x, y = c * tw / pw, r * th / ph
                if not (border <= x < tw - border and border <= y < th - border):
                    continue
                gx, gy = c0 + x, r0 + y
                if padding <= gx < W - padding and padding <= gy < H - padding:
                    pts.append({"col": gx, "row": gy, "base_size": sigma})
            # density integrates to a count; rescale so the stitched map keeps that meaning
            up = np.array(Image.fromarray(dm).resize((tw, th), Image.BILINEAR), np.float32)
            up *= (ph * pw) / float(th * tw)
            wt = cosine_weight(th, tw)
            heat[r0 : r0 + th, c0 : c0 + tw] += up * wt
            wsum[r0 : r0 + th, c0 : c0 + tw] += wt
    heat = np.where(wsum > 0, heat / wsum, 0).astype(np.float32)
    df = merge_points(
        pd.DataFrame(pts, columns=["col", "row", "base_size"]), merge_radius, x="col", y="row"
    )
    if len(df):
        xy = df[["col", "row"]].to_numpy(float)
        k = min(len(df), 3)
        if k > 1:
            dist, _ = cKDTree(xy).query(xy, k=k)
            local = 0.8 * dist[:, 1:k].mean(1) * 2
        else:
            local = np.full(len(df), np.nan)
        base = df["base_size"].to_numpy(float)
        base = np.where(np.isfinite(base), base, local)
        loc = np.where(np.isfinite(local), local, base)
        df["size_px"] = 0.5 * base + 0.5 * loc
        xs = da.x.values
        ys = da.y.values
        df["x"] = np.interp(df["col"] - 0.5, np.arange(W), xs)
        df["y"] = np.interp(df["row"] - 0.5, np.arange(H), ys)
        res = abs(float(xs[1] - xs[0])) if W > 1 else 1.0
        df["size_m"] = df["size_px"] * res if da.attrs.get("units_xy") != "px" else np.nan
    else:
        df = pd.DataFrame(
            columns=["col", "row", "base_size", "n_merged", "size_px", "x", "y", "size_m"]
        )
    dens = xr.DataArray(
        heat,
        dims=("y", "x"),
        coords={"y": da.y, "x": da.x},
        name="density",
        attrs={"count_integral": float(heat.sum())},
    )
    if da.rio.crs is not None:
        dens = dens.rio.write_crs(da.rio.crs)
    df = df.rename(columns={"base_size": "model_size_px"})
    return df.reset_index(drop=True), dens


def size_classes(
    points: pd.DataFrame, col: str = "size_px", q: tuple[float, float] = (25, 75)
) -> pd.Series:
    """Label plants ``small`` / ``medium`` / ``large`` by size quartiles (LUMS colour classes)."""
    lo, hi = np.nanpercentile(points[col], q) if len(points) else (np.nan, np.nan)
    return pd.Series(
        np.select([points[col] < lo, points[col] > hi], ["small", "large"], "medium"),
        index=points.index,
        name="size_class",
    )
