"""Remote sensing foundation models: catalog, loaders, raster embeddings and probes.

One entry point for the geospatial foundation models (GFMs) in use today. The catalog covers
multispectral optical encoders, multi-sensor and any-modality models, pixel time series and
global embedding products, very high resolution models, vision-language models and
weather/Earth system emulators. Models with public weights load through one call,
:func:`load_model`, which dispatches to TorchGeo, TerraTorch, open_clip, ``torch.hub`` or
Hugging Face ``transformers``. Each loaded model is wrapped in :class:`FoundationEncoder`, so
any of them turns an image batch or a georeferenced raster into embeddings the same way.

Quick start::

    import geolattice.rsense.foundation_models as fm

    fm.list_models(category="multimodal")              # what is available
    fm.recommend("segmentation", sensor="sentinel-2")  # which model to try first

    enc = fm.load_model("dofa-base", bands=["blue", "green", "red", "nir"])
    emb = enc.embed_raster(s2_da, chip_size=224)        # (embedding, y, x) DataArray
    sim = fm.similarity_map(emb, emb.isel(y=3, x=5))    # where else looks like this chip?
    fm.linear_probe(X, y, cv=5)                          # how much do the embeddings know?

    img = fm.alphaearth_image(aoi, 2024)                 # Google AlphaEarth, in Earth Engine
    df = fm.sample_alphaearth(points, 2024)              # 64 embedding columns per point

Heavy dependencies (PyTorch, TorchGeo, TerraTorch, open_clip, Earth Engine) are imported only
when a function needs them: ``pip install "geolattice[rsense-foundation]"``. Weights download on first
use into the TorchGeo / Hugging Face caches. Licenses and preprocessing change between
releases, so check each model card before you rely on a model.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
import xarray as xr

# =============================================================================================
# 1. Catalog
# =============================================================================================

CATEGORIES = {
    "optical": "Multispectral optical encoders (Sentinel-2, Landsat, HLS)",
    "multimodal": "Multi-sensor and any-modality encoders (optical, SAR, DEM, any band set)",
    "timeseries": "Pixel time series models and precomputed global embedding products",
    "vhr": "Very high resolution (aerial, sub-metre satellite) encoders and segmenters",
    "vision-language": "Image-text models for retrieval, zero-shot classification and chat",
    "weather": "Weather and Earth system emulators on gridded fields",
}

LOADERS = {
    "torchgeo": "torchgeo.models (weights from the TorchGeo hub)",
    "terratorch": "terratorch BACKBONE_REGISTRY",
    "open_clip": "open_clip with a Hugging Face checkpoint",
    "torch.hub": "torch.hub.load from the authors' repository",
    "transformers": "Hugging Face transformers AutoModel",
    "earthengine": "Google Earth Engine image collection (no local model)",
    "adapter": "geolattice.rsense.dl.foundation adapter (needs a local checkpoint or extra package)",
    "external": "authors' code only; listed for reference",
}


@dataclass(frozen=True)
class FoundationModel:
    """Catalog entry for one foundation model (or one released size of it)."""

    name: str
    category: str
    provider: str
    year: int
    architecture: str
    modalities: tuple[str, ...]
    inputs: str
    temporal: bool
    embed_dim: int | None
    params: str
    weights: str
    loader: str
    target: str | None = None
    use_cases: tuple[str, ...] = ()
    reference: str = ""
    notes: str = ""
    defaults: Mapping[str, Any] = field(default_factory=dict)

    @property
    def loadable(self) -> bool:
        """True when :func:`load_model` can build the model with downloadable weights."""
        return self.loader in ("torchgeo", "terratorch", "open_clip", "torch.hub", "transformers")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["loadable"] = self.loadable
        return d


_S2_10 = ("blue", "green", "red", "rededge1", "rededge2", "rededge3", "nir", "nir08", "swir16",
          "swir22")
_HLS = ("blue", "green", "red", "nir08", "swir16", "swir22")
_PRITHVI_BANDS = ("BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2")


def _m(**kw: Any) -> FoundationModel:
    kw["modalities"] = tuple(kw.get("modalities", ()))
    kw["use_cases"] = tuple(kw.get("use_cases", ()))
    return FoundationModel(**kw)


_ENTRIES: list[FoundationModel] = [
    # --- multispectral optical ------------------------------------------------------------
    _m(name="prithvi-eo-1.0-100m", category="optical", provider="NASA / IBM", year=2023,
       architecture="ViT-B MAE, 3-D patch embedding", modalities=["optical"],
       inputs="HLS: blue, green, red, nir08, swir16, swir22 (30 m)", temporal=True, embed_dim=768,
       params="100M", weights="ibm-nasa-geospatial/Prithvi-EO-1.0-100M", loader="terratorch",
       target="terratorch_prithvi_eo_v1_100",
       use_cases=["flood mapping", "burn scars", "crop segmentation"],
       reference="Jakubik et al. 2023, arXiv:2310.18660",
       defaults={"bands": _PRITHVI_BANDS, "img_size": 224}),
    _m(name="prithvi-eo-2.0-300m", category="optical", provider="NASA / IBM", year=2024,
       architecture="ViT-L MAE, 3-D patch embedding", modalities=["optical"],
       inputs="HLS 6 bands, 224 px, 1 to 4 time steps", temporal=True, embed_dim=1024,
       params="300M", weights="ibm-nasa-geospatial/Prithvi-EO-2.0-300M", loader="terratorch",
       target="terratorch_prithvi_eo_v2_300",
       use_cases=["flood mapping", "burn scars", "crop segmentation", "land cover"],
       reference="Szwarcman et al. 2024, arXiv:2412.02732",
       defaults={"bands": _PRITHVI_BANDS, "img_size": 224}),
    _m(name="prithvi-eo-2.0-300m-tl", category="optical", provider="NASA / IBM", year=2024,
       architecture="ViT-L MAE with time and location encodings", modalities=["optical"],
       inputs="HLS 6 bands + acquisition date + lat/lon", temporal=True, embed_dim=1024,
       params="300M", weights="ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL", loader="terratorch",
       target="terratorch_prithvi_eo_v2_300_tl", use_cases=["multi-temporal crop mapping"],
       reference="Szwarcman et al. 2024", defaults={"bands": _PRITHVI_BANDS, "img_size": 224}),
    _m(name="prithvi-eo-2.0-600m", category="optical", provider="NASA / IBM", year=2024,
       architecture="ViT-H MAE, 3-D patch embedding", modalities=["optical"],
       inputs="HLS 6 bands", temporal=True, embed_dim=1280, params="600M",
       weights="ibm-nasa-geospatial/Prithvi-EO-2.0-600M", loader="terratorch",
       target="terratorch_prithvi_eo_v2_600", use_cases=["segmentation", "regression"],
       reference="Szwarcman et al. 2024", defaults={"bands": _PRITHVI_BANDS, "img_size": 224}),
    _m(name="satmae", category="optical", provider="Stanford", year=2022,
       architecture="ViT MAE with temporal and spectral group encodings", modalities=["optical"],
       inputs="fMoW RGB or Sentinel-2 10 bands", temporal=True, embed_dim=1024, params="300M",
       weights="authors' checkpoints (github.com/sustainlab-group/SatMAE)", loader="adapter",
       target="geolattice.rsense.dl.foundation.SatMAE", use_cases=["scene classification"],
       reference="Cong et al. 2022, NeurIPS"),
    _m(name="scale-mae", category="optical", provider="UC Berkeley", year=2023,
       architecture="ViT-L MAE with GSD-aware positional encoding", modalities=["optical"],
       inputs="RGB at any ground sample distance", temporal=False, embed_dim=1024,
       params="300M", weights="torchgeo ScaleMAELarge16_Weights.FMOW_RGB", loader="torchgeo",
       target="scalemae_large_patch16:ScaleMAELarge16_Weights.FMOW_RGB",
       use_cases=["multi-resolution scene classification"],
       reference="Reed et al. 2023, ICCV", defaults={"img_size": 224}),
    _m(name="ssl4eo-s12-resnet18-moco", category="optical", provider="TUM / DLR", year=2022,
       architecture="ResNet-18, MoCo-v2", modalities=["optical"],
       inputs="Sentinel-2 L1C 13 bands, reflectance / 10000", temporal=False, embed_dim=512,
       params="11M", weights="torchgeo ResNet18_Weights.SENTINEL2_ALL_MOCO", loader="torchgeo",
       target="resnet18:ResNet18_Weights.SENTINEL2_ALL_MOCO",
       use_cases=["lightweight transfer learning", "embeddings on CPU"],
       reference="Wang et al. 2023, IEEE GRSM", defaults={"img_size": 224}),
    _m(name="ssl4eo-s12-resnet50-moco", category="optical", provider="TUM / DLR", year=2022,
       architecture="ResNet-50, MoCo-v2", modalities=["optical"],
       inputs="Sentinel-2 L1C 13 bands", temporal=False, embed_dim=2048, params="25M",
       weights="torchgeo ResNet50_Weights.SENTINEL2_ALL_MOCO", loader="torchgeo",
       target="resnet50:ResNet50_Weights.SENTINEL2_ALL_MOCO", use_cases=["transfer learning"],
       reference="Wang et al. 2023", defaults={"img_size": 224}),
    _m(name="ssl4eo-s12-vits16-dino", category="optical", provider="TUM / DLR", year=2022,
       architecture="ViT-S/16, DINO", modalities=["optical"], inputs="Sentinel-2 L1C 13 bands",
       temporal=False, embed_dim=384, params="22M",
       weights="torchgeo ViTSmall16_Weights.SENTINEL2_ALL_DINO", loader="torchgeo",
       target="vit_small_patch16_224:ViTSmall16_Weights.SENTINEL2_ALL_DINO",
       use_cases=["transfer learning"], reference="Wang et al. 2023", defaults={"img_size": 224}),
    _m(name="ssl4eo-s1-resnet50-moco", category="optical", provider="TUM / DLR", year=2022,
       architecture="ResNet-50, MoCo-v2", modalities=["sar"],
       inputs="Sentinel-1 GRD VV, VH (dB)", temporal=False, embed_dim=2048, params="25M",
       weights="torchgeo ResNet50_Weights.SENTINEL1_ALL_MOCO", loader="torchgeo",
       target="resnet50:ResNet50_Weights.SENTINEL1_ALL_MOCO", use_cases=["SAR transfer learning"],
       reference="Wang et al. 2023", defaults={"img_size": 224}),
    _m(name="satlas-swint-s2-ms", category="optical", provider="Allen Institute for AI",
       year=2023, architecture="Swin-v2-T", modalities=["optical"],
       inputs="Sentinel-2 9 bands, reflectance / 3000 clipped to [0, 1]", temporal=False,
       embed_dim=768, params="28M", weights="torchgeo Swin_V2_T_Weights.SENTINEL2_SI_MS_SATLAS",
       loader="torchgeo", target="swin_v2_t:Swin_V2_T_Weights.SENTINEL2_SI_MS_SATLAS",
       use_cases=["land cover", "infrastructure", "solar farms"],
       reference="Bastani et al. 2023, ICCV", defaults={"img_size": 256}),
    _m(name="satlas-swinb-s2-ms", category="optical", provider="Allen Institute for AI",
       year=2023, architecture="Swin-v2-B", modalities=["optical"], inputs="Sentinel-2 9 bands",
       temporal=False, embed_dim=1024, params="88M",
       weights="torchgeo Swin_V2_B_Weights.SENTINEL2_SI_MS_SATLAS", loader="torchgeo",
       target="swin_v2_b:Swin_V2_B_Weights.SENTINEL2_SI_MS_SATLAS", use_cases=["land cover"],
       reference="Bastani et al. 2023", defaults={"img_size": 256}),
    _m(name="satlas-swinb-naip", category="vhr", provider="Allen Institute for AI", year=2023,
       architecture="Swin-v2-B", modalities=["optical"], inputs="NAIP RGB (1 m)",
       temporal=False, embed_dim=1024, params="88M",
       weights="torchgeo Swin_V2_B_Weights.NAIP_RGB_SI_SATLAS", loader="torchgeo",
       target="swin_v2_b:Swin_V2_B_Weights.NAIP_RGB_SI_SATLAS", use_cases=["aerial mapping"],
       reference="Bastani et al. 2023", defaults={"img_size": 256}),
    _m(name="spectralgpt", category="optical", provider="Wuhan University et al.", year=2024,
       architecture="3-D spectral-spatial MAE", modalities=["optical", "hyperspectral"],
       inputs="Sentinel-2 12 bands, 3-D tokens", temporal=False, embed_dim=768, params="100M",
       weights="authors' checkpoints (github.com/danfenghong/IEEE_TPAMI_SpectralGPT)",
       loader="external", use_cases=["spectral classification", "change detection"],
       reference="Hong et al. 2024, IEEE TPAMI"),
    # --- multi-sensor and any-modality ----------------------------------------------------
    _m(name="clay-v1.5", category="multimodal", provider="Clay Foundation", year=2024,
       architecture="ViT-L MAE, wavelength-aware patch embedding",
       modalities=["optical", "sar"], inputs="any sensor: pixels + wavelengths + GSD + time + lat/lon",
       temporal=True, embed_dim=1024, params="311M", weights="made-with-clay/Clay",
       loader="adapter", target="geolattice.rsense.dl.foundation.Clay",
       use_cases=["embeddings", "similarity search", "land cover"],
       reference="Clay Foundation 2024, clay-foundation.github.io/model",
       notes="Needs the claymodel package; TerraTorch also registers timm_clay_v1_base"),
    _m(name="dofa-base", category="multimodal", provider="TUM / Zhu lab", year=2024,
       architecture="ViT-B/16 with a wavelength-conditioned dynamic stem",
       modalities=["optical", "sar", "hyperspectral"], inputs="any band set + band wavelengths",
       temporal=False, embed_dim=768, params="86M", weights="torchgeo DOFABase16_Weights.DOFA_MAE",
       loader="torchgeo", target="dofa_base_patch16_224:DOFABase16_Weights.DOFA_MAE",
       use_cases=["mixed sensors", "unusual band sets", "hyperspectral"],
       reference="Xiong et al. 2024, arXiv:2403.15356", defaults={"img_size": 224}),
    _m(name="dofa-large", category="multimodal", provider="TUM / Zhu lab", year=2024,
       architecture="ViT-L/16 with a wavelength-conditioned dynamic stem",
       modalities=["optical", "sar", "hyperspectral"], inputs="any band set + band wavelengths",
       temporal=False, embed_dim=1024, params="300M",
       weights="torchgeo DOFALarge16_Weights.DOFA_MAE", loader="torchgeo",
       target="dofa_large_patch16_224:DOFALarge16_Weights.DOFA_MAE",
       use_cases=["mixed sensors"], reference="Xiong et al. 2024", defaults={"img_size": 224}),
    _m(name="croma-base", category="multimodal", provider="University of Waterloo", year=2023,
       architecture="Two ViT-B encoders, contrastive + MAE, cross-modal fusion",
       modalities=["optical", "sar"], inputs="Sentinel-2 12 bands and/or Sentinel-1 VV, VH",
       temporal=False, embed_dim=768, params="2 x 86M",
       weights="torchgeo CROMABase_Weights.CROMA_VIT", loader="torchgeo",
       target="croma_base:CROMABase_Weights.CROMA_VIT",
       use_cases=["optical-SAR fusion", "cloudy regions"],
       reference="Fuller et al. 2023, NeurIPS", defaults={"img_size": 120}),
    _m(name="copernicus-fm", category="multimodal", provider="TUM / Zhu lab", year=2025,
       architecture="ViT-B with dynamic hypernetwork patch embedding + metadata encoding",
       modalities=["optical", "sar", "atmosphere", "dem"],
       inputs="any Sentinel 1/2/3/5P band set + wavelengths, bandwidths, lon/lat/time/area",
       temporal=True, embed_dim=768, params="139M",
       weights="torchgeo CopernicusFM_Base_Weights.CopernicusFM_ViT", loader="torchgeo",
       target="copernicusfm_base:CopernicusFM_Base_Weights.CopernicusFM_ViT",
       use_cases=["Copernicus multi-mission", "atmospheric variables"],
       reference="Wang et al. 2025, arXiv:2503.11849", defaults={"img_size": 224}),
    _m(name="panopticon", category="multimodal", provider="Panopticon team", year=2025,
       architecture="ViT-B/14 DINOv2 with spectral cross-attention patch embedding",
       modalities=["optical", "sar", "hyperspectral"], inputs="any channels + wavelength ids (nm)",
       temporal=False, embed_dim=768, params="86M",
       weights="torchgeo Panopticon_Weights.VIT_BASE14", loader="torchgeo",
       target="panopticon_vitb14:Panopticon_Weights.VIT_BASE14",
       use_cases=["any sensor embeddings"], reference="Waldmann et al. 2025, arXiv:2503.10845",
       defaults={"img_size": 224}),
    _m(name="terramind-v1-base", category="multimodal", provider="IBM / ESA", year=2025,
       architecture="Any-to-any generative transformer (tokens + pixels)",
       modalities=["optical", "sar", "dem", "landcover", "ndvi", "text"],
       inputs="S2L2A, S1GRD/S1RTC, DEM, LULC, NDVI (choose with modalities=)", temporal=False,
       embed_dim=768, params="~90M encoder", weights="ibm-esa-geospatial/TerraMind-1.0-base",
       loader="terratorch", target="terratorch_terramind_v1_base",
       use_cases=["missing modality synthesis", "multimodal segmentation"],
       reference="Jakubik et al. 2025, arXiv:2504.11171",
       defaults={"modalities": ("S2L2A",), "img_size": 224}),
    _m(name="terramind-v1-large", category="multimodal", provider="IBM / ESA", year=2025,
       architecture="Any-to-any generative transformer", modalities=["optical", "sar", "dem"],
       inputs="as terramind-v1-base", temporal=False, embed_dim=1024, params="~300M encoder",
       weights="ibm-esa-geospatial/TerraMind-1.0-large", loader="terratorch",
       target="terratorch_terramind_v1_large", use_cases=["multimodal segmentation"],
       reference="Jakubik et al. 2025", defaults={"modalities": ("S2L2A",), "img_size": 224}),
    _m(name="anysat", category="multimodal", provider="IGN / ENPC (LASTIG)", year=2025,
       architecture="JEPA with scale-adaptive patch encoders",
       modalities=["optical", "sar", "vhr", "timeseries"],
       inputs="dict of modalities at any resolution (aerial, S1, S2, Landsat, SPOT, ...)",
       temporal=True, embed_dim=768, params="125M", weights="github.com/gastruc/AnySat",
       loader="torch.hub", target="gastruc/anysat:anysat",
       use_cases=["heterogeneous datasets", "crop type", "land cover"],
       reference="Astruc et al. 2025, CVPR",
       notes="forward takes a dict of modality tensors and patch_size= (metres)"),
    _m(name="galileo", category="multimodal", provider="NASA Harvest / Ai2 et al.", year=2025,
       architecture="ViT with global + local masked objectives over space-time-channel tokens",
       modalities=["optical", "sar", "dem", "weather", "timeseries"],
       inputs="S1, S2, ERA5, TerraClimate, DEM, VIIRS, ... as pixels or patches", temporal=True,
       embed_dim=768, params="nano to base", weights="nasaharvest/galileo",
       loader="external", use_cases=["crop mapping", "time series", "small-object detection"],
       reference="Tseng et al. 2025, arXiv:2502.09356"),
    _m(name="olmoearth", category="multimodal", provider="Allen Institute for AI", year=2025,
       architecture="Multimodal spatiotemporal ViT family", modalities=["optical", "sar", "timeseries"],
       inputs="Sentinel-1, Sentinel-2, Landsat time series", temporal=True, embed_dim=None,
       params="nano to large", weights="allenai/OlmoEarth (Hugging Face)", loader="external",
       use_cases=["land cover", "crop mapping", "forest loss"],
       reference="Ai2 2025, olmoearth.allenai.org"),
    _m(name="skysense", category="multimodal", provider="Ant Group / Wuhan University", year=2024,
       architecture="Factorized multimodal spatiotemporal encoder",
       modalities=["vhr", "optical", "sar", "timeseries"],
       inputs="high-res optical + Sentinel-2 time series + Sentinel-1", temporal=True,
       embed_dim=None, params="2B", weights="released on request", loader="external",
       use_cases=["benchmarks", "multimodal segmentation"], reference="Guo et al. 2024, CVPR"),
    # --- pixel time series and embedding products -----------------------------------------
    _m(name="alphaearth", category="timeseries", provider="Google DeepMind", year=2025,
       architecture="Space-time-precision embedding field model",
       modalities=["optical", "sar", "lidar", "climate", "timeseries"],
       inputs="annual 10 m embeddings, bands A00 to A63 (unit length)", temporal=True,
       embed_dim=64, params="n/a (precomputed)",
       weights="Earth Engine GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL", loader="earthengine",
       target="GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL",
       use_cases=["covariates for spatial ML", "digital soil mapping", "change detection",
                  "similarity search"],
       reference="Brown et al. 2025, arXiv:2507.22291",
       notes="Use alphaearth_image / sample_alphaearth / alphaearth_to_xarray"),
    _m(name="presto", category="timeseries", provider="NASA Harvest", year=2023,
       architecture="Lightweight transformer on pixel time series",
       modalities=["optical", "sar", "climate", "dem", "timeseries"],
       inputs="monthly S1, S2, ERA5, SRTM, NDVI per pixel", temporal=True, embed_dim=128,
       params="0.4M", weights="github.com/nasaharvest/presto", loader="external",
       use_cases=["crop type mapping", "cheap CPU inference"],
       reference="Tseng et al. 2023, arXiv:2304.14065"),
    _m(name="tessera", category="timeseries", provider="University of Cambridge", year=2025,
       architecture="Dual-branch (S1, S2) pixel time series encoder, Barlow Twins",
       modalities=["optical", "sar", "timeseries"], inputs="annual S1 + S2 pixel time series",
       temporal=True, embed_dim=128, params="n/a", weights="github.com/ucam-eo/tessera",
       loader="external", use_cases=["precomputed 10 m embeddings", "crop and habitat mapping"],
       reference="Feng et al. 2025, arXiv:2506.20380"),
    # --- very high resolution --------------------------------------------------------------
    _m(name="dinov3-sat-vitl16", category="vhr", provider="Meta", year=2025,
       architecture="ViT-L/16 distilled from the 7B DINOv3 satellite model",
       modalities=["optical"], inputs="RGB Maxar-like imagery (SAT-493M)", temporal=False,
       embed_dim=1024, params="300M", weights="facebook/dinov3-vitl16-pretrain-sat493m",
       loader="transformers", target="facebook/dinov3-vitl16-pretrain-sat493m",
       use_cases=["canopy height", "dense features", "object mapping"],
       reference="Simeoni et al. 2025, arXiv:2508.10104",
       notes="Gated on Hugging Face: accept the license and log in with `hf auth login`",
       defaults={"img_size": 224}),
    _m(name="sam-vit-h", category="vhr", provider="Meta", year=2023,
       architecture="ViT-H image encoder + prompt decoder", modalities=["optical"],
       inputs="RGB, any size (resized to 1024)", temporal=False, embed_dim=256, params="636M",
       weights="segment-anything sam_vit_h_4b8939.pth", loader="adapter",
       target="geolattice.rsense.dl.foundation.SAM", use_cases=["zero-shot object delineation"],
       reference="Kirillov et al. 2023, ICCV", notes="Runs through segment-geospatial"),
    _m(name="sam2-hiera-large", category="vhr", provider="Meta", year=2024,
       architecture="Hiera-L image/video encoder + memory attention", modalities=["optical"],
       inputs="RGB images or image sequences", temporal=True, embed_dim=256, params="224M",
       weights="facebook/sam2.1-hiera-large", loader="adapter",
       target="geolattice.rsense.classify.image_classification.sam_segment",
       use_cases=["object delineation", "field boundaries", "tracking"],
       reference="Ravi et al. 2024, arXiv:2408.00714", notes="Use samgeo.SamGeo2"),
    # --- vision-language -------------------------------------------------------------------
    _m(name="remoteclip-vit-b-32", category="vision-language", provider="Liu et al.", year=2024,
       architecture="CLIP ViT-B/32 tuned on remote sensing captions",
       modalities=["optical", "text"], inputs="RGB 224 px + text", temporal=False, embed_dim=512,
       params="151M", weights="chendelong/RemoteCLIP", loader="open_clip",
       target="remoteclip-vit-b-32", use_cases=["zero-shot scene classification", "retrieval"],
       reference="Liu et al. 2024, IEEE TGRS"),
    _m(name="remoteclip-vit-l-14", category="vision-language", provider="Liu et al.", year=2024,
       architecture="CLIP ViT-L/14", modalities=["optical", "text"], inputs="RGB 224 px + text",
       temporal=False, embed_dim=768, params="428M", weights="chendelong/RemoteCLIP",
       loader="open_clip", target="remoteclip-vit-l-14", use_cases=["retrieval"],
       reference="Liu et al. 2024"),
    _m(name="georsclip-vit-b-32", category="vision-language", provider="Zhang et al.", year=2024,
       architecture="CLIP ViT-B/32 tuned on RS5M", modalities=["optical", "text"],
       inputs="RGB 224 px + text", temporal=False, embed_dim=512, params="151M",
       weights="Zilun/GeoRSCLIP", loader="open_clip", target="georsclip-vit-b-32",
       use_cases=["zero-shot classification", "retrieval"],
       reference="Zhang et al. 2024, IEEE TGRS (RS5M)"),
    _m(name="skyclip", category="vision-language", provider="Wang et al.", year=2024,
       architecture="CLIP tuned on SkyScript", modalities=["optical", "text"],
       inputs="RGB + text", temporal=False, embed_dim=768, params="428M",
       weights="github.com/wangzhecheng/SkyScript", loader="external",
       use_cases=["fine-grained zero-shot classification"], reference="Wang et al. 2024, AAAI"),
    _m(name="geochat", category="vision-language", provider="MBZUAI", year=2024,
       architecture="LLaVA-1.5 with a remote sensing instruction set",
       modalities=["optical", "text"], inputs="RGB + chat prompt", temporal=False,
       embed_dim=None, params="7B", weights="MBZUAI/geochat-7B", loader="external",
       use_cases=["visual question answering", "grounded description"],
       reference="Kuckreja et al. 2024, CVPR"),
    _m(name="earthgpt", category="vision-language", provider="Zhang et al.", year=2024,
       architecture="Multimodal LLM over optical, SAR and infrared",
       modalities=["optical", "sar", "infrared", "text"], inputs="image + prompt",
       temporal=False, embed_dim=None, params="13B", weights="authors' release",
       loader="external", use_cases=["captioning", "VQA", "detection"],
       reference="Zhang et al. 2024, IEEE TGRS"),
    _m(name="lhrs-bot", category="vision-language", provider="Nanjing University", year=2024,
       architecture="Multimodal LLM trained on VGI-aligned imagery", modalities=["optical", "text"],
       inputs="image + prompt", temporal=False, embed_dim=None, params="7B",
       weights="github.com/NJU-LHRS/LHRS-Bot", loader="external",
       use_cases=["VQA", "scene reasoning"], reference="Muhtar et al. 2024, ECCV"),
    _m(name="earthdial", category="vision-language", provider="MBZUAI", year=2025,
       architecture="InternVL-based multi-sensor, multi-temporal assistant",
       modalities=["optical", "sar", "infrared", "timeseries", "text"], inputs="image(s) + prompt",
       temporal=True, embed_dim=None, params="4B", weights="akshaydudhane/EarthDial_4B_RGB",
       loader="external", use_cases=["change description", "multispectral VQA"],
       reference="Soni et al. 2025, CVPR"),
    # --- weather and Earth system ----------------------------------------------------------
    _m(name="prithvi-wxc", category="weather", provider="NASA / IBM", year=2024,
       architecture="2.3B transformer on MERRA-2", modalities=["weather"],
       inputs="MERRA-2 surface and pressure-level fields", temporal=True, embed_dim=None,
       params="2.3B", weights="ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M", loader="external",
       use_cases=["downscaling", "gravity waves", "forecast fine-tuning"],
       reference="Schmude et al. 2024, arXiv:2409.13598"),
    _m(name="aurora", category="weather", provider="Microsoft", year=2024,
       architecture="3-D Swin U-Net with Perceiver encoders", modalities=["weather"],
       inputs="ERA5/HRES fields at 0.25 or 0.1 degrees", temporal=True, embed_dim=None,
       params="1.3B", weights="microsoft/aurora (torchgeo Aurora_Weights)", loader="external",
       use_cases=["weather", "air pollution", "ocean waves"],
       reference="Bodnar et al. 2025, Nature", notes="pip install microsoft-aurora"),
    _m(name="graphcast", category="weather", provider="Google DeepMind", year=2023,
       architecture="Graph neural network on an icosahedral mesh", modalities=["weather"],
       inputs="ERA5 0.25 degree fields", temporal=True, embed_dim=None, params="37M",
       weights="github.com/google-deepmind/graphcast", loader="external",
       use_cases=["10-day forecasts"], reference="Lam et al. 2023, Science"),
    _m(name="gencast", category="weather", provider="Google DeepMind", year=2024,
       architecture="Diffusion model on an icosahedral mesh", modalities=["weather"],
       inputs="ERA5 0.25 degree fields", temporal=True, embed_dim=None, params="n/a",
       weights="github.com/google-deepmind/graphcast", loader="external",
       use_cases=["probabilistic ensemble forecasts"], reference="Price et al. 2025, Nature"),
    _m(name="fourcastnet", category="weather", provider="NVIDIA", year=2022,
       architecture="Adaptive Fourier neural operator (v1), spherical FNO (v2)",
       modalities=["weather"], inputs="ERA5 0.25 degree fields", temporal=True, embed_dim=None,
       params="n/a", weights="NVIDIA earth2studio / modulus", loader="external",
       use_cases=["fast forecasts", "large ensembles"], reference="Pathak et al. 2022, arXiv:2202.11214"),
]

#: All catalog entries by name.
MODELS: dict[str, FoundationModel] = {m.name: m for m in _ENTRIES}

#: Short aliases accepted by :func:`get_model_spec` and :func:`load_model`.
ALIASES = {
    "prithvi": "prithvi-eo-2.0-300m", "prithvi-v1": "prithvi-eo-1.0-100m",
    "prithvi-v2": "prithvi-eo-2.0-300m", "clay": "clay-v1.5", "dofa": "dofa-base",
    "croma": "croma-base", "copernicusfm": "copernicus-fm", "terramind": "terramind-v1-base",
    "scalemae": "scale-mae", "ssl4eo": "ssl4eo-s12-resnet18-moco", "satlas": "satlas-swint-s2-ms",
    "dinov3-sat": "dinov3-sat-vitl16", "remoteclip": "remoteclip-vit-b-32",
    "georsclip": "georsclip-vit-b-32", "sam": "sam-vit-h", "sam2": "sam2-hiera-large",
    "alphaearth-foundations": "alphaearth", "satellite-embedding": "alphaearth",
}


def get_model_spec(name: str) -> FoundationModel:
    """Catalog entry for ``name`` (case-insensitive; aliases such as ``"dofa"`` work)."""
    key = name.lower().strip()
    key = ALIASES.get(key, key)
    if key not in MODELS:
        close = [m for m in MODELS if key.split("-")[0] in m]
        hint = f"; did you mean {close}?" if close else "; see list_models()"
        raise KeyError(f"unknown foundation model {name!r}{hint}")
    return MODELS[key]


def model_info(name: str) -> dict[str, Any]:
    """All catalog fields for one model, as a dict."""
    return get_model_spec(name).to_dict()


def list_models(category: str | None = None, *, modality: str | None = None,
                loadable: bool | None = None, temporal: bool | None = None) -> pd.DataFrame:
    """Table of the catalog, one row per model, optionally filtered.

    ``category`` is a key of :data:`CATEGORIES`; ``modality`` keeps models that accept it
    (``"optical"``, ``"sar"``, ``"text"``, ...); ``loadable=True`` keeps models that
    :func:`load_model` can build.
    """
    if category is not None and category not in CATEGORIES:
        raise ValueError(f"category must be one of {sorted(CATEGORIES)}")
    rows = []
    for m in _ENTRIES:
        if category and m.category != category:
            continue
        if modality and modality not in m.modalities:
            continue
        if loadable is not None and m.loadable != loadable:
            continue
        if temporal is not None and m.temporal != temporal:
            continue
        rows.append({"name": m.name, "category": m.category, "provider": m.provider,
                     "year": m.year, "modalities": ", ".join(m.modalities),
                     "temporal": m.temporal, "embed_dim": m.embed_dim, "params": m.params,
                     "loader": m.loader, "loadable": m.loadable, "weights": m.weights,
                     "use_cases": "; ".join(m.use_cases)})
    return pd.DataFrame(rows).set_index("name") if rows else pd.DataFrame()


_RECOMMEND = {
    "covariates": ["alphaearth", "clay-v1.5", "presto", "ssl4eo-s12-resnet18-moco"],
    "segmentation": ["prithvi-eo-2.0-300m", "terramind-v1-base", "satlas-swinb-s2-ms",
                     "dofa-base"],
    "classification": ["ssl4eo-s12-resnet50-moco", "dofa-base", "scale-mae",
                       "remoteclip-vit-b-32"],
    "timeseries": ["presto", "galileo", "prithvi-eo-2.0-300m-tl", "anysat", "alphaearth"],
    "multisensor": ["dofa-base", "copernicus-fm", "panopticon", "terramind-v1-base",
                    "croma-base", "clay-v1.5"],
    "sar": ["croma-base", "ssl4eo-s1-resnet50-moco", "dofa-base", "terramind-v1-base"],
    "hyperspectral": ["dofa-base", "panopticon", "spectralgpt"],
    "vhr": ["dinov3-sat-vitl16", "satlas-swinb-naip", "sam2-hiera-large", "sam-vit-h"],
    "zero-shot": ["remoteclip-vit-b-32", "georsclip-vit-b-32", "skyclip", "sam2-hiera-large"],
    "similarity": ["alphaearth", "clay-v1.5", "dofa-base", "dinov3-sat-vitl16"],
    "change": ["alphaearth", "prithvi-eo-2.0-300m", "dofa-base"],
    "weather": ["aurora", "prithvi-wxc", "graphcast", "gencast", "fourcastnet"],
}
_TASK_SYNONYMS = {
    "soil": "covariates", "digital soil mapping": "covariates", "dsm": "covariates",
    "regression": "covariates", "embeddings": "covariates", "spatial ml": "covariates",
    "crop": "timeseries", "phenology": "timeseries", "land cover": "segmentation",
    "flood": "segmentation", "burn": "segmentation", "detection": "vhr", "object": "vhr",
    "retrieval": "zero-shot", "search": "similarity", "change detection": "change",
}
_SENSOR_MODALITY = {"sentinel-1": "sar", "sar": "sar", "sentinel-2": "optical",
                    "landsat": "optical", "hls": "optical", "naip": "optical",
                    "planet": "optical", "maxar": "optical", "hyperspectral": "hyperspectral",
                    "emit": "hyperspectral", "enmap": "hyperspectral", "era5": "weather"}


def recommend(task: str, *, sensor: str | None = None, loadable_only: bool = False
              ) -> pd.DataFrame:
    """Models to try first for ``task``, best first.

    ``task`` is one of ``covariates``, ``segmentation``, ``classification``, ``timeseries``,
    ``multisensor``, ``sar``, ``hyperspectral``, ``vhr``, ``zero-shot``, ``similarity``,
    ``change`` or ``weather`` (common synonyms such as ``"soil"`` or ``"crop"`` work).
    ``sensor`` (``"sentinel-1"``, ``"landsat"``, ``"emit"``, ...) drops models that do not
    accept that modality. The order is a starting point from benchmark results and practice,
    not a guarantee: always compare against a supervised baseline on your own data.
    """
    t = task.lower().strip()
    t = _TASK_SYNONYMS.get(t, t)
    if t not in _RECOMMEND:
        raise ValueError(f"task must be one of {sorted(_RECOMMEND)} (or a synonym)")
    names = _RECOMMEND[t]
    if sensor:
        mod = _SENSOR_MODALITY.get(sensor.lower(), sensor.lower())
        names = [n for n in names if mod in MODELS[n].modalities
                 or (mod == "optical" and "vhr" in MODELS[n].modalities)]
    if loadable_only:
        names = [n for n in names if MODELS[n].loadable or MODELS[n].loader == "earthengine"]
    rows = [{"rank": i + 1, "name": n, "loader": MODELS[n].loader,
             "why": "; ".join(MODELS[n].use_cases)} for i, n in enumerate(names)]
    return pd.DataFrame(rows).set_index("rank") if rows else pd.DataFrame()


# =============================================================================================
# 2. Band wavelengths and input preparation
# =============================================================================================

#: Central wavelengths (nm) of GeoLattice canonical band names, Sentinel-2 values where sensors
#: differ. Used for wavelength-aware models (DOFA, Copernicus-FM, Panopticon, Clay).
WAVELENGTHS_NM: dict[str, float] = {
    "coastal": 443.0, "blue": 490.0, "green": 560.0, "red": 665.0, "rededge1": 705.0,
    "rededge2": 740.0, "rededge3": 783.0, "nir": 842.0, "nir08": 865.0, "watervapor": 945.0,
    "cirrus": 1375.0, "swir16": 1610.0, "swir22": 2190.0, "pan": 640.0,
    "thermal": 10900.0, "lwir11": 10900.0, "lwir12": 12000.0,
}

#: Bandwidths (nm) for the same bands (Sentinel-2 MSI), needed by Copernicus-FM.
BANDWIDTHS_NM: dict[str, float] = {
    "coastal": 20.0, "blue": 65.0, "green": 35.0, "red": 30.0, "rededge1": 15.0,
    "rededge2": 15.0, "rededge3": 20.0, "nir": 115.0, "nir08": 20.0, "watervapor": 20.0,
    "cirrus": 30.0, "swir16": 90.0, "swir22": 180.0, "pan": 170.0,
}

#: Panopticon channel ids for SAR polarizations (negative integers by convention).
SAR_CHANNEL_IDS = {"vv": -1, "vh": -2, "hh": -3, "hv": -4}

_ALIAS_BANDS = {"b01": "coastal", "b02": "blue", "b03": "green", "b04": "red",
                "b05": "rededge1", "b06": "rededge2", "b07": "rededge3", "b08": "nir",
                "b8a": "nir08", "b09": "watervapor", "b10": "cirrus", "b11": "swir16",
                "b12": "swir22", "nir_narrow": "nir08", "swir_1": "swir16", "swir_2": "swir22"}


def band_wavelengths(bands: Sequence[str], *, unit: Literal["nm", "um"] = "nm") -> list[float]:
    """Central wavelengths for band names (``"red"``, ``"B04"``, ``"swir16"``, ...)."""
    out = []
    for b in bands:
        k = _ALIAS_BANDS.get(str(b).lower(), str(b).lower())
        if k not in WAVELENGTHS_NM:
            raise KeyError(f"no wavelength for band {b!r}; pass wavelengths= explicitly")
        out.append(WAVELENGTHS_NM[k])
    return [w / 1000.0 for w in out] if unit == "um" else out


def normalize(x: np.ndarray, *, method: Literal["standard", "scale", "minmax", "none"] = "standard",
              mean: Sequence[float] | None = None, std: Sequence[float] | None = None,
              scale: float = 10000.0, clip: tuple[float, float] | None = None,
              channel_axis: int = 1) -> np.ndarray:
    """Normalize an image batch before it goes into a model.

    ``standard`` subtracts ``mean`` and divides by ``std`` per channel (computed from ``x``
    when not given; pass the model card statistics for best results). ``scale`` divides by
    ``scale`` (for example 10000 for Sentinel-2 L2A, 3000 for Satlas). ``minmax`` rescales
    each channel to [0, 1]. NaNs become 0 after normalization.
    """
    a = np.asarray(x, dtype="float32")
    ax = channel_axis % a.ndim
    others = tuple(i for i in range(a.ndim) if i != ax)
    shape = [1] * a.ndim
    shape[ax] = a.shape[ax]
    if method == "standard":
        mu = np.asarray(mean, "float32") if mean is not None else np.nanmean(a, axis=others)
        sd = np.asarray(std, "float32") if std is not None else np.nanstd(a, axis=others)
        a = (a - mu.reshape(shape)) / np.maximum(sd.reshape(shape), 1e-6)
    elif method == "scale":
        a = a / float(scale)
    elif method == "minmax":
        lo = np.nanmin(a, axis=others).reshape(shape)
        hi = np.nanmax(a, axis=others).reshape(shape)
        a = (a - lo) / np.maximum(hi - lo, 1e-6)
    elif method != "none":
        raise ValueError("method must be 'standard', 'scale', 'minmax' or 'none'")
    if clip is not None:
        a = np.clip(a, *clip)
    return np.nan_to_num(a, nan=0.0).astype("float32", copy=False)


# ---- model input contracts --------------------------------------------------------------------

_S2_13 = ["coastal", "blue", "green", "red", "rededge1", "rededge2", "rededge3", "nir", "nir08",
          "watervapor", "cirrus", "swir16", "swir22"]
_S2_12 = [b for b in _S2_13 if b != "cirrus"]
_HLS6 = ["blue", "green", "red", "nir08", "swir16", "swir22"]
_IMAGENET = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

#: What each loadable model expects: band order (GeoLattice names), the units the statistics refer to
#: (``dn``: reflectance x 10000, ``reflectance``: 0 to 1, ``db``: SAR backscatter, ``unit``:
#: values already scaled to 0 to 1), and the mean / std applied after scaling. ``None`` means the
#: model has no published statistics and inputs are standardized per band. Sources: TorchGeo
#: weight transforms, TerraTorch pretraining constants and the model cards.
INPUT_SPECS: dict[str, dict[str, Any]] = {
    "prithvi-eo-1.0-100m": {"bands": _HLS6, "units": "dn", "img_size": 224,
                            "mean": [775.0, 1081.0, 1229.0, 2497.0, 2204.0, 1611.0],
                            "std": [1282.0, 1270.0, 1399.0, 1368.0, 1292.0, 1155.0]},
    "prithvi-eo-2.0": {"bands": _HLS6, "units": "dn", "img_size": 224,
                       "mean": [1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0],
                       "std": [2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0]},
    "ssl4eo-s12": {"bands": _S2_13, "units": "dn", "img_size": 224,
                   "mean": [0.0] * 13, "std": [10000.0] * 13},
    "ssl4eo-s1": {"bands": ["vv", "vh"], "units": "db", "img_size": 224,
                  "mean": [-12.59, -20.26], "std": [5.26, 5.91]},
    "satlas-swin": {"bands": ["red", "green", "blue", "rededge1", "rededge2", "rededge3", "nir",
                              "swir16", "swir22"], "units": "dn", "img_size": 256,
                    "mean": [0.0] * 9, "std": [3000.0] * 3 + [8160.0] * 6, "clip": (0.0, 1.0),
                    "note": "RGB come from the 8-bit TCI product (/255); DN/3000 approximates it"},
    "satlas-swinb-naip": {"bands": ["red", "green", "blue"], "units": "unit", "img_size": 256,
                          "mean": [0.0] * 3, "std": [1.0] * 3},
    "scale-mae": {"bands": ["red", "green", "blue"], "units": "unit", "img_size": 224,
                  "mean": _IMAGENET[0], "std": _IMAGENET[1]},
    "terramind-v1": {"bands": _S2_12, "units": "dn", "img_size": 224,
                     "mean": [1390.458, 1503.317, 1718.197, 1853.91, 2199.1, 2779.975, 2987.011,
                              3083.234, 3132.22, 3162.988, 2424.884, 1857.648],
                     "std": [2106.761, 2141.107, 2038.973, 2134.138, 2085.321, 1889.926,
                             1820.257, 1871.918, 1753.829, 1797.379, 1434.261, 1334.311],
                     "S1GRD": {"bands": ["vv", "vh"], "units": "db", "mean": [-12.599, -20.293],
                               "std": [5.195, 5.890]}},
    "croma-base": {"bands": [b for b in _S2_13 if b != "cirrus"], "units": "dn", "img_size": 120,
                   "mean": None, "std": None,
                   "sar": {"bands": ["vv", "vh"], "units": "db", "mean": None, "std": None}},
    "dofa": {"bands": None, "units": "any", "img_size": 224, "mean": None, "std": None},
    "copernicus-fm": {"bands": None, "units": "any", "img_size": 224, "mean": None, "std": None},
    "panopticon": {"bands": None, "units": "any", "img_size": 224, "mean": None, "std": None},
    "dinov3-sat": {"bands": ["red", "green", "blue"], "units": "unit", "img_size": 224,
                   "mean": _IMAGENET[0], "std": _IMAGENET[1]},
}


def input_spec(name: str) -> dict[str, Any]:
    """The input contract of a model: band order, units, statistics and chip size.

    ``bands`` is ``None`` for wavelength-aware models (DOFA, Copernicus-FM, Panopticon), which
    accept any band set; ``mean`` / ``std`` of ``None`` mean per-band standardization.
    """
    n = get_model_spec(name).name
    for key in sorted(INPUT_SPECS, key=len, reverse=True):
        if n.startswith(key) or n == key:
            return dict(INPUT_SPECS[key])
    if n.startswith("satlas-") and "-s2-ms" in n:
        return dict(INPUT_SPECS["satlas-swin"])
    raise KeyError(f"no input specification recorded for {n!r}")


def prepare(name: str, da: xr.DataArray, *, units: Literal["auto", "reflectance", "dn", "db",
                                                             "unit"] = "auto",
            missing: Literal["zero", "nearest", "error"] = "nearest",
            modality: str | None = None) -> xr.DataArray:
    """Turn a raster into the input a model expects: band order, scaling and normalization.

    ``da`` is ``(band, y, x)`` or ``(time, band, y, x)`` with GeoLattice band names. ``units`` says
    how its values are stored (``auto``: reflectance when the 99th percentile is below 2, DN
    otherwise). Bands the model needs but the raster lacks are filled from the band nearest in
    wavelength (``missing="nearest"``), with zeros after normalization (``"zero"``), or raise an
    error. For models without published statistics every band is standardized over the raster.
    ``modality`` picks a secondary input (``"S1GRD"`` for TerraMind, ``"sar"`` for CROMA).
    Returns float32 with the model's band order in the ``band`` coordinate.
    """
    spec = input_spec(name)
    if modality:
        spec = {**spec, **spec[modality]}
    bd = next(d for d in da.dims if d not in ("y", "x", "time"))
    have = [str(b) for b in da[bd].values]
    a = da.astype("float32")
    if units == "auto":
        if spec["units"] == "db":
            units = "db"
        else:
            q = float(np.nanpercentile(a.values[np.isfinite(a.values)], 99)) if a.size else 0.0
            units = "reflectance" if q < 2 else "dn"
    want = spec["bands"] or have
    filled = []
    layers = []
    for b in want:
        k = _ALIAS_BANDS.get(b.lower(), b.lower())
        src = next((h for h in have if _ALIAS_BANDS.get(h.lower(), h.lower()) == k), None)
        if src is None:
            if missing == "error":
                raise KeyError(f"{name} needs band {b!r}, which the raster does not have")
            filled.append(b)
            if missing == "nearest" and k in WAVELENGTHS_NM:
                cand = [h for h in have if _ALIAS_BANDS.get(h.lower(), h.lower()) in WAVELENGTHS_NM]
                src = min(cand, key=lambda h: abs(WAVELENGTHS_NM[_ALIAS_BANDS.get(
                    h.lower(), h.lower())] - WAVELENGTHS_NM[k])) if cand else None
        layers.append(a.sel({bd: src}) if src is not None else xr.zeros_like(a.isel({bd: 0})) * np.nan)
    x = xr.concat(layers, dim=pd.Index(list(want), name="band"))
    if bd != "band":
        x = x.drop_vars(bd, errors="ignore")
    target = spec["units"]
    if target == "dn" and units == "reflectance":
        x = x * 10000.0
    elif target in ("reflectance", "unit") and units == "dn":
        x = x / 10000.0
    if target == "unit" and units in ("reflectance", "dn"):
        x = (x / 0.3).clip(0, 1)  # reflectance to an 8-bit-like 0..1 display range
    if spec.get("mean") is not None:
        mu = xr.DataArray(np.asarray(spec["mean"], "float32"), dims="band", coords={"band": x.band})
        sd = xr.DataArray(np.asarray(spec["std"], "float32"), dims="band", coords={"band": x.band})
        x = (x - mu) / sd
    else:
        dims = [d for d in x.dims if d != "band"]
        x = (x - x.mean(dims)) / x.std(dims).clip(min=1e-6)
    if spec.get("clip"):
        x = x.clip(*spec["clip"])
    x = x.fillna(0.0).astype("float32")
    if missing == "zero":
        for b in filled:
            x.loc[{"band": b}] = 0.0
    x = x.transpose(*(["time"] if "time" in x.dims else []), "band", "y", "x")
    x.attrs.update(model=get_model_spec(name).name, input_units=units, filled_bands=",".join(filled))
    try:
        if da.rio.crs is not None:
            x = x.rio.write_crs(da.rio.crs)
    except Exception:
        pass
    return x


def chip_raster(da: xr.DataArray, *, chip_size: int = 224, stride: int | None = None,
                band_dim: str | None = None, time_dim: str | None = "time", pad: bool = True
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cut a ``(band, y, x)`` raster, or a ``(time, band, y, x)`` series, into square chips.

    Returns ``(chips, ys, xs)``: chips of shape ``(n, band, chip, chip)``, or
    ``(n, band, time, chip, chip)`` for a series (the layout Prithvi expects), and the y / x
    coordinates of each chip centre in row-major order. ``stride`` defaults to ``chip_size``
    (no overlap). With ``pad=True`` the right and bottom edges are padded with the reflected
    image so every pixel is covered.
    """
    stride = stride or chip_size
    has_time = time_dim is not None and time_dim in da.dims
    if band_dim is None:
        band_dim = next(d for d in da.dims if d not in ("y", "x", time_dim)) \
            if has_time else _band_dim(da)
    order = (band_dim, time_dim, "y", "x") if has_time else (band_dim, "y", "x")
    a = da.transpose(*order).values.astype("float32")
    ny, nx = a.shape[-2:]
    lead = [(0, 0)] * (a.ndim - 2)
    if pad:
        py = max(chip_size - ny, 0) + (-(max(ny, chip_size) - chip_size) % stride)
        px = max(chip_size - nx, 0) + (-(max(nx, chip_size) - chip_size) % stride)
        if py or px:
            mode = "reflect" if (py < ny and px < nx) else "edge"
            a = np.pad(a, lead + [(0, py), (0, px)], mode=mode)
    Ny, Nx = a.shape[-2:]
    if Ny < chip_size or Nx < chip_size:
        raise ValueError(f"raster {ny}x{nx} is smaller than chip_size={chip_size}; use pad=True")
    rows = range(0, Ny - chip_size + 1, stride)
    cols = range(0, Nx - chip_size + 1, stride)
    chips = np.stack([a[..., r:r + chip_size, c:c + chip_size] for r in rows for c in cols])
    yc, xc = _centre_coords(da, rows, chip_size, "y", ny), _centre_coords(da, cols, chip_size,
                                                                           "x", nx)
    return chips, yc, xc


def _centre_coords(da, starts, chip, dim, n):
    v = np.asarray(da[dim].values, dtype="float64")
    step = (v[-1] - v[0]) / (n - 1) if n > 1 else 1.0
    return np.array([v[0] + step * (s + (chip - 1) / 2.0) for s in starts])


def _band_dim(da: xr.DataArray) -> str:
    for d in da.dims:
        if d not in ("y", "x"):
            return d
    raise ValueError("raster needs a band dimension besides y and x")


def chip_metadata(ys: Sequence[float], xs: Sequence[float], crs: Any, *,
                  time: Any = None, chip_size: int | None = None,
                  resolution: float | None = None) -> pd.DataFrame:
    """Longitude, latitude, date and area of every chip, in the order of :func:`chip_raster`.

    ``ys`` and ``xs`` are chip centres in ``crs``; ``time`` is one date for all chips or one
    per chip. Columns: ``lon``, ``lat`` (degrees), ``days`` (since 1970-01-01, NaN without a
    date), ``year``, ``doy`` (day of year) and ``area_km2`` (from ``chip_size`` pixels of
    ``resolution`` metres, NaN when not given). This is the metadata Copernicus-FM takes
    (``lon, lat, days, area``) and the temporal / location coordinates of Prithvi-TL.
    """
    from pyproj import Transformer

    yy, xx = np.meshgrid(np.asarray(ys, "float64"), np.asarray(xs, "float64"), indexing="ij")
    tr = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon, lat = tr.transform(xx.ravel(), yy.ravel())
    n = lon.size
    if time is None:
        t = pd.to_datetime(pd.Series([pd.NaT] * n))
    else:
        t = pd.to_datetime(pd.Series(np.broadcast_to(np.asarray(time, dtype="datetime64[ns]"),
                                                     (n,))))
    days = (t - pd.Timestamp("1970-01-01")).dt.total_seconds() / 86400.0
    area = (chip_size * resolution / 1000.0) ** 2 if chip_size and resolution else np.nan
    return pd.DataFrame({"lon": lon, "lat": lat, "days": days.to_numpy(),
                         "year": t.dt.year.to_numpy(dtype="float64", na_value=np.nan),
                         "doy": t.dt.dayofyear.to_numpy(dtype="float64", na_value=np.nan),
                         "area_km2": np.full(n, area)})


# =============================================================================================
# 3. Loading models
# =============================================================================================


def _require(module: str, extra: str = "rsense-foundation"):
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as e:
        try:
            from ..install_options import install_hint

            hint = install_hint(extra)
        except Exception:  # pragma: no cover - metadata problems
            hint = f'pip install "geolattice[{extra}]"'
        raise ImportError(f"{module} is needed here: {hint}") from e


def _torch():
    return _require("torch")


def pool_features(out: Any, *, pool: Literal["mean", "cls", "none"] = "mean"):
    """Reduce model output to one vector per image.

    Accepts a tensor ``(B, D)``, tokens ``(B, N, D)``, a feature map ``(B, C, H, W)``, a list
    of those (the last is used) or a dict (``"pooled"``, ``"embedding"`` or the first tensor).
    ``pool="cls"`` takes token 0, ``"mean"`` averages tokens or pixels, ``"none"`` returns
    the tensor unchanged.
    """
    if isinstance(out, Mapping):
        for k in ("pooled", "pooler_output", "embedding", "embeddings", "joint_GAP",
                  "optical_GAP", "sar_GAP", "last_hidden_state"):
            if k in out and out[k] is not None:
                out = out[k]
                break
        else:
            out = next(v for v in out.values() if hasattr(v, "shape"))
    elif hasattr(out, "pooler_output") and getattr(out, "pooler_output", None) is not None:
        out = out.pooler_output
    elif hasattr(out, "last_hidden_state"):
        out = out.last_hidden_state
    if isinstance(out, (list, tuple)):
        out = out[-1]
    if pool == "none" or out.ndim == 2:
        return out
    if out.ndim == 3:
        return out[:, 0] if pool == "cls" else out.mean(1)
    if out.ndim == 4:
        return out.mean((2, 3))
    if out.ndim == 5:  # (B, C, T, H, W)
        return out.mean((2, 3, 4))
    raise ValueError(f"cannot pool output of shape {tuple(out.shape)}")


class FoundationEncoder:
    """A loaded foundation model with one interface for every backend.

    ``encode(x)`` takes a ``(B, C, H, W)`` tensor or array and returns ``(B, D)``
    embeddings; ``embed(x)`` does the same in batches and returns NumPy;
    ``embed_raster(da)`` tiles a georeferenced raster and returns an embedding map.
    ``model`` is the underlying ``torch.nn.Module`` for fine-tuning.
    """

    def __init__(self, spec: FoundationModel, model: Any, *, device: Any = None,
                 forward: Callable[..., Any] | None = None, bands: Sequence[str] | None = None,
                 wavelengths: Sequence[float] | None = None, img_size: int | None = None,
                 pool: Literal["mean", "cls"] = "mean", extras: Mapping[str, Any] | None = None):
        self.spec = spec
        self.name = spec.name
        self.model = model
        self.device = device
        self.bands = list(bands) if bands else None
        self.wavelengths = list(wavelengths) if wavelengths else None
        self.img_size = img_size or spec.defaults.get("img_size", 224)
        self.pool = pool
        self.extras = dict(extras or {})
        self._forward = forward or (lambda m, x, **kw: m(x))

    def __repr__(self) -> str:
        return (f"FoundationEncoder({self.name!r}, loader={self.spec.loader!r}, "
                f"embed_dim={self.spec.embed_dim}, device={self.device})")

    def to(self, device: Any) -> FoundationEncoder:
        torch = _torch()
        self.device = torch.device(device)
        self.model.to(self.device)
        return self

    def forward_features(self, x: Any, **kwargs: Any) -> Any:
        """Raw model output (tokens, feature maps or a dict) without pooling."""
        torch = _torch()
        if isinstance(x, Mapping):  # multi-modality input (AnySat, TerraMind)
            x = {k: (torch.as_tensor(v).to(self.device) if not isinstance(v, (str, list))
                     else v) for k, v in x.items()}
            x = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v)
                 for k, v in x.items()}
        else:
            if not torch.is_tensor(x):
                x = torch.as_tensor(np.asarray(x, dtype="float32"))
            x = x.to(self.device, dtype=torch.float32)
        with torch.inference_mode():
            return self._forward(self.model, x, enc=self, **kwargs)

    def encode(self, x: Any, *, pool: Literal["mean", "cls", "none"] | None = None,
               **kwargs: Any):
        """Embeddings for a batch ``(B, C, H, W)``; returns a torch tensor ``(B, D)``."""
        return pool_features(self.forward_features(x, **kwargs), pool=pool or self.pool)

    def embed(self, x: Any, *, batch_size: int = 16, normalize_embeddings: bool = False,
              **kwargs: Any) -> np.ndarray:
        """Batched :meth:`encode` returning a ``float32`` NumPy array ``(B, D)``."""
        a = np.asarray(x, dtype="float32")
        if a.ndim == 3:
            a = a[None]
        outs = []
        for i in range(0, len(a), batch_size):
            kw = _slice_batch_kwargs(kwargs, i, batch_size, len(a))
            e = self.encode(a[i:i + batch_size], **kw)
            outs.append(e.float().cpu().numpy())
        E = np.concatenate(outs, 0) if outs else np.zeros((0, self.spec.embed_dim or 0), "float32")
        if normalize_embeddings:
            E = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
        return E.astype("float32", copy=False)

    def dense_features(self, x: Any, **kwargs: Any) -> np.ndarray:
        """Per-patch features ``(B, D, h, w)`` as NumPy, before pooling.

        Works for models whose output is a feature map (ResNet, Satlas Swin) or patch tokens
        (Prithvi, TerraMind, Scale-MAE, SSL4EO ViT, DINOv3, DOFA). Class and register tokens
        are dropped; for time series the tokens are averaged over time steps.
        """
        torch = _torch()
        m = self.model
        if self.spec.name.startswith("dofa"):
            out = self._dofa_tokens(x, **kwargs)
        elif hasattr(m, "features") and hasattr(m, "head") and isinstance(
                m.head, torch.nn.Identity):  # torchvision Swin: (B, H, W, C) map
            t = torch.as_tensor(np.asarray(x, "float32")).to(self.device)
            with torch.inference_mode():
                out = m.features(t).permute(0, 3, 1, 2)
        else:
            out = self.forward_features(x, **kwargs)
            if isinstance(out, Mapping) or hasattr(out, "last_hidden_state"):
                out = out["last_hidden_state"] if isinstance(out, Mapping) else out.last_hidden_state
            if isinstance(out, (list, tuple)):
                out = out[-1]
        steps = np.shape(x)[2] if np.ndim(x) == 5 else 1  # time steps of a (B, C, T, H, W) series
        return _tokens_to_grid(out, steps=steps).float().cpu().numpy()

    def _dofa_tokens(self, x, wavelengths=None, **kw):
        torch = _torch()
        m = self.model
        wl = wavelengths or self.wavelengths
        if wl is None:
            raise ValueError("DOFA needs band wavelengths: pass bands= or wavelengths=")
        wl = [w / 1000.0 if w > 100 else w for w in wl]
        t = torch.as_tensor(np.asarray(x, "float32")).to(self.device)
        with torch.inference_mode():
            tok, _ = m.patch_embed(t, torch.tensor(wl, device=t.device).float())
            tok = tok + m.pos_embed[:, 1:, :]
            cls = (m.cls_token + m.pos_embed[:, :1, :]).expand(tok.shape[0], -1, -1)
            tok = torch.cat((cls, tok), 1)
            for blk in m.blocks:
                tok = blk(tok)
            norm = m.fc_norm if getattr(m, "global_pool", False) else m.norm
            return norm(tok[:, 1:, :])

    def embed_raster(self, da: xr.DataArray, *, chip_size: int | None = None,
                     stride: int | None = None, batch_size: int = 16,
                     norm: Mapping[str, Any] | str | None = None, resize: bool = True,
                     dense: bool = False, metadata: bool | str = False,
                     **kwargs: Any) -> xr.DataArray:
        """Tile a ``(band, y, x)`` raster, or a ``(time, band, y, x)`` series, and embed it.

        Returns a DataArray ``(embedding, y, x)`` with the raster CRS kept. By default there
        is one cell per chip, centred on the chip. With ``dense=True`` every chip contributes
        its grid of patch features (for a ViT with 16 px patches on 224 px chips, 14 x 14
        cells per chip), stitched into one map; this needs ``stride`` equal to the chip size.

        ``norm`` holds keyword arguments for :func:`normalize` (default: per-band
        standardization over the raster), or ``"model"`` to apply :func:`prepare`, the
        model's own band order and published statistics, before chipping. With ``resize=True`` chips are resized to the model
        input size when ``chip_size`` differs from it. ``metadata=True`` builds per-chip
        location and date inputs with :func:`chip_metadata`: the ``metadata`` tensor of
        Copernicus-FM, or ``temporal_coords`` and ``location_coords`` for Prithvi.
        """
        chip = chip_size or self.img_size
        if isinstance(norm, str):
            if norm != "model":
                raise ValueError("norm must be a dict of normalize() arguments or 'model'")
            da = prepare(self.name, da)
            norm = {"method": "none"}
        tdim = "time" if "time" in da.dims else None
        bd = next(d for d in da.dims if d not in ("y", "x", tdim))
        if self.bands is None and bd in da.coords:
            names = [str(b) for b in da[bd].values]
            optical = all(_ALIAS_BANDS.get(n.lower(), n.lower()) in WAVELENGTHS_NM for n in names)
            if optical or all(n.lower() in SAR_CHANNEL_IDS for n in names):
                self.bands = names
                if optical and self.wavelengths is None and self.spec.name.startswith(
                        ("dofa", "copernicus", "panopticon")):
                    self.wavelengths = band_wavelengths(names)
        if dense and stride not in (None, chip):
            raise ValueError("dense=True needs stride equal to chip_size")
        chips, ys, xs = chip_raster(da, chip_size=chip, stride=stride, band_dim=bd,
                                    time_dim=tdim)
        chips = normalize(chips, **(dict(norm) if norm else {}))
        size = self.img_size if resize and chip != self.img_size else None  # resized batch by batch
        if metadata:
            kwargs.update(self._metadata_kwargs(da, ys, xs, chip, tdim))
        attrs = {"model": self.name, "chip_size": chip, "stride": stride or chip,
                 "dense": int(bool(dense)), "normalization": str(dict(norm) if norm else
                                                                 {"method": "standard"})}
        if not dense:
            parts = []
            for i in range(0, len(chips), batch_size):
                c = chips[i:i + batch_size]
                kw = _slice_batch_kwargs(kwargs, i, batch_size, len(chips))
                parts.append(self.embed(_resize(c, size) if size else c, batch_size=batch_size, **kw))
            E = np.concatenate(parts, 0)
            E = E.reshape(len(ys), len(xs), -1).transpose(2, 0, 1)
            out = xr.DataArray(E, dims=("embedding", "y", "x"),
                               coords={"embedding": np.arange(E.shape[0]), "y": ys, "x": xs},
                               name=f"{self.name}_embedding", attrs=attrs)
        else:
            grids = []
            for i in range(0, len(chips), batch_size):
                kw = _slice_batch_kwargs(kwargs, i, batch_size, len(chips))
                c = chips[i:i + batch_size]
                grids.append(self.dense_features(_resize(c, size) if size else c, **kw))
            G = np.concatenate(grids, 0)  # (n, D, h, w)
            n, D, h, w = G.shape
            G = G.reshape(len(ys), len(xs), D, h, w).transpose(2, 0, 3, 1, 4)
            G = G.reshape(D, len(ys) * h, len(xs) * w)
            yv, xv = np.asarray(da["y"].values, "float64"), np.asarray(da["x"].values, "float64")
            dy = (yv[-1] - yv[0]) / (len(yv) - 1) if len(yv) > 1 else -1.0
            dx = (xv[-1] - xv[0]) / (len(xv) - 1) if len(xv) > 1 else 1.0
            cy, cx = chip / h, chip / w  # input pixels per feature cell
            yc = yv[0] + dy * ((np.arange(G.shape[1]) + 0.5) * cy - 0.5)
            xc = xv[0] + dx * ((np.arange(G.shape[2]) + 0.5) * cx - 0.5)
            keep_y = int(np.ceil(len(yv) / cy))
            keep_x = int(np.ceil(len(xv) / cx))
            out = xr.DataArray(G[:, :keep_y, :keep_x], dims=("embedding", "y", "x"),
                               coords={"embedding": np.arange(D), "y": yc[:keep_y],
                                       "x": xc[:keep_x]},
                               name=f"{self.name}_dense", attrs=attrs)
        try:
            if da.rio.crs is not None:
                out = out.rio.write_crs(da.rio.crs)
        except Exception:  # rioxarray accessor missing or no CRS
            pass
        return out

    def _metadata_kwargs(self, da, ys, xs, chip, tdim):
        torch = _torch()
        try:
            crs = da.rio.crs
        except Exception:
            crs = None
        if crs is None:
            raise ValueError("metadata=True needs a raster with a CRS (da.rio.write_crs)")
        res = abs(float(da["x"].values[1] - da["x"].values[0])) if da.sizes["x"] > 1 else None
        times = da[tdim].values if tdim else da.attrs.get("time")
        t0 = None if times is None else np.atleast_1d(times)[0]
        meta = chip_metadata(ys, xs, crs, time=t0, chip_size=chip, resolution=res)
        if self.spec.name == "copernicus-fm":
            return {"metadata": torch.as_tensor(np.ascontiguousarray(
                meta[["lon", "lat", "days", "area_km2"]].to_numpy("float32")))}
        if self.spec.name.startswith("prithvi"):
            n = len(meta)
            loc = torch.as_tensor(np.ascontiguousarray(meta[["lat", "lon"]].to_numpy("float32")))
            if times is None:
                return {"location_coords": loc}
            tt = pd.to_datetime(np.atleast_1d(times))
            tc = np.stack([tt.year.to_numpy("float32"), (tt.dayofyear - 1).to_numpy("float32")], 1)
            return {"location_coords": loc,
                    "temporal_coords": torch.as_tensor(np.broadcast_to(tc, (n,) + tc.shape).copy())}
        return {}


def _slice_batch_kwargs(kwargs: Mapping[str, Any], i: int, bs: int, n: int) -> dict[str, Any]:
    """Slice per-sample keyword arguments (metadata tensors) to the current batch."""
    out = {}
    for k, v in kwargs.items():
        if k in ("metadata", "temporal_coords", "location_coords", "x_sar") and \
                hasattr(v, "shape") and len(v) == n:
            out[k] = v[i:i + bs]
        else:
            out[k] = v
    return out


def _tokens_to_grid(out, steps: int | None = None):
    """``(B, N, D)`` tokens or ``(B, D, h, w)`` maps to a ``(B, D, h, w)`` grid.

    ``steps`` is the number of time steps behind the tokens when known; otherwise it is inferred,
    which is ambiguous for some token counts (785 = 1 + 4 x 14 x 14 = 1 + 28 x 28).
    """
    torch = _torch()
    if not torch.is_tensor(out):
        out = torch.as_tensor(np.asarray(out))
    if out.ndim == 4:
        return out
    if out.ndim == 5:  # (B, D, T, h, w)
        return out.mean(2)
    if out.ndim == 2:
        raise ValueError("this model returns one pooled vector per chip, not patch features; "
                         "use dense=False")
    if out.ndim != 3:
        raise ValueError(f"cannot turn output of shape {tuple(out.shape)} into a grid")
    B, N, D = out.shape
    for t in ((steps,) if steps else (1, 2, 3, 4, 6, 8, 12)):  # time steps, then prefix tokens
        for prefix in (0, 1, 5, 4):
            g = (N - prefix) / t
            r = int(round(g ** 0.5))
            if g == int(g) and r * r == int(g) and r > 1:
                tok = out[:, prefix:, :].reshape(B, t, r, r, D).mean(1)
                return tok.permute(0, 3, 1, 2)
    raise ValueError(f"cannot infer a square patch grid from {N} tokens")


def _resize(chips: np.ndarray, size: int) -> np.ndarray:
    torch = _torch()
    t = torch.as_tensor(chips)
    lead = t.shape[:-2]
    t = t.reshape(-1, 1 if t.ndim == 3 else int(np.prod(lead[1:])), *t.shape[-2:])
    t = torch.nn.functional.interpolate(t, size=(size, size), mode="bilinear",
                                        align_corners=False)
    return t.reshape(*lead, size, size).numpy()


def _resolve_device(device: Any):
    torch = _torch()
    if device is not None:
        return torch.device(device)
    try:
        from .dl._torch import torch_device

        return torch_device()
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- per-backend forward functions -----------------------------------------------------------

def _fw_plain(m, x, **kw):
    if hasattr(m, "forward_features"):
        return m.forward_features(x)
    return m(x)


def _fw_dofa(m, x, *, enc, wavelengths=None, **kw):
    wl = wavelengths or enc.wavelengths
    if wl is None:
        raise ValueError("DOFA needs band wavelengths: load_model(..., bands=[...]) or "
                         "wavelengths=[...] in µm")
    wl = [w / 1000.0 if w > 100 else w for w in wl]  # accept nm or µm
    return m.forward_features(x, wl)


def _fw_croma(m, x, *, enc, x_sar=None, **kw):
    if enc.extras.get("modality") == "sar":
        return m(x_sar=x)
    if x_sar is not None:
        x_sar = _as_device(x_sar, x)
    return m(x_optical=x, x_sar=x_sar)


def _fw_copernicusfm(m, x, *, enc, metadata=None, wavelengths=None, bandwidths=None, **kw):
    torch = _torch()
    wl = wavelengths or enc.wavelengths or (band_wavelengths(enc.bands) if enc.bands else None)
    if wl is None:
        raise ValueError("Copernicus-FM needs band wavelengths (nm): pass bands= or wavelengths=")
    wl = [w * 1000.0 if w < 100 else w for w in wl]
    bw = bandwidths or enc.extras.get("bandwidths")
    if bw is None and enc.bands:
        bw = [BANDWIDTHS_NM.get(_ALIAS_BANDS.get(b.lower(), b.lower()), 30.0) for b in enc.bands]
    bw = bw or [30.0] * len(wl)
    if metadata is None:
        metadata = torch.full((x.shape[0], 4), float("nan"), device=x.device)
    elif not torch.is_tensor(metadata):
        metadata = torch.as_tensor(np.asarray(metadata, "float32"), device=x.device)
    return m.forward_features(x, metadata.to(x.device), wl, bw)


def _fw_panopticon(m, x, *, enc, chn_ids=None, **kw):
    torch = _torch()
    ids = chn_ids
    if ids is None:
        if enc.bands:
            ids = [SAR_CHANNEL_IDS[b.lower()] if b.lower() in SAR_CHANNEL_IDS
                   else band_wavelengths([b])[0] for b in enc.bands]
        elif enc.wavelengths:
            ids = [w * 1000.0 if w < 100 else w for w in enc.wavelengths]
        else:
            raise ValueError("Panopticon needs bands= or wavelengths= (nm)")
    ids = torch.as_tensor(ids, dtype=torch.float32, device=x.device)
    if ids.ndim == 1:
        ids = ids.expand(x.shape[0], -1)
    return m({"imgs": x, "chn_ids": ids})


def _fw_terratorch(m, x, *, enc, temporal_coords=None, location_coords=None, **kw):
    mods = enc.extras.get("modalities")
    if mods and not isinstance(x, Mapping):
        if len(mods) > 1:
            raise ValueError(f"pass a dict of tensors keyed by modality: {mods}")
        x = {mods[0]: x}
    coords = {}
    if temporal_coords is not None:
        coords["temporal_coords"] = _as_device(temporal_coords, x)
    if location_coords is not None:
        coords["location_coords"] = _as_device(location_coords, x)
    return m(x, **coords) if coords else m(x)


def _as_device(v, x):
    torch = _torch()
    ref = next(iter(x.values())) if isinstance(x, Mapping) else x
    return torch.as_tensor(np.asarray(v, "float32") if not torch.is_tensor(v) else v
                           ).to(ref.device, dtype=torch.float32)


def _fw_clip(m, x, **kw):
    return m.encode_image(x)


def _fw_hf(m, x, **kw):
    return m(pixel_values=x)


def _fw_anysat(m, x, *, enc, patch_size=None, modality=None, output="tile", **kw):
    mod = modality or enc.extras.get("modality", "s2")
    batch = x if isinstance(x, Mapping) else {mod: x}
    return m(batch, patch_size=patch_size or enc.extras.get("patch_size", 10), output=output)


def _load_torchgeo(spec, bands, wavelengths, **kwargs):
    tg = _require("torchgeo.models")
    fn_name, weight = spec.target.split(":")
    enum_name, member = weight.split(".")
    w = getattr(getattr(tg, enum_name), member) if kwargs.pop("pretrained", True) else None
    fn = getattr(tg, fn_name)
    forward = _fw_plain
    extras: dict[str, Any] = {}
    n = spec.name
    if n.startswith("dofa"):
        model = fn(weights=w, **kwargs)
        forward = _fw_dofa
        if wavelengths is None and bands:
            wavelengths = band_wavelengths(bands)
    elif n.startswith("croma"):
        modality = kwargs.pop("modality", "optical")
        model = fn(weights=w, modalities=["sar", "optical"] if modality == "both"
                   else [modality], **kwargs)
        forward, extras["modality"] = _fw_croma, modality
    elif n == "copernicus-fm":
        model, forward = fn(weights=w, **kwargs), _fw_copernicusfm
    elif n == "panopticon":
        model, forward = fn(weights=w, **kwargs), _fw_panopticon
    else:
        model = fn(weights=w, **kwargs)
        torch = _torch()
        if hasattr(model, "reset_classifier"):  # timm ResNet / ViT
            model.reset_classifier(0)
        elif hasattr(model, "head") and isinstance(model.head, torch.nn.Linear):  # torchvision Swin
            model.head = torch.nn.Identity()
            forward = lambda m, x, **kw: m(x)  # noqa: E731
    if w is not None:
        model.rs_weights_meta = dict(getattr(w, "meta", {}))
    return model, forward, wavelengths, extras


def _load_terratorch(spec, bands, wavelengths, **kwargs):
    reg = _require("terratorch.registry")
    kw = {"pretrained": kwargs.pop("pretrained", True)}
    extras: dict[str, Any] = {}
    if spec.name.startswith("prithvi"):
        kw["bands"] = [b.upper() for b in bands] if bands else list(spec.defaults["bands"])
        kw["num_frames"] = kwargs.pop("num_frames", 1)
    if spec.name.startswith("terramind"):
        mods = list(kwargs.pop("modalities", spec.defaults.get("modalities", ("S2L2A",))))
        kw["modalities"], extras["modalities"] = mods, mods
    kw.update(kwargs)
    model = reg.BACKBONE_REGISTRY.build(spec.target, **kw)
    return model, _fw_terratorch, wavelengths, extras


def _load_open_clip(spec, **kwargs):
    from .classify.image_classification import load_rs_clip

    model, preprocess, tokenizer = load_rs_clip(spec.target, device="cpu")
    return model, _fw_clip, {"preprocess": preprocess, "tokenizer": tokenizer}


def _load_hub(spec, **kwargs):
    torch = _torch()
    repo, entry = spec.target.split(":")
    kwargs.setdefault("pretrained", True)
    if spec.name == "anysat":
        kwargs.setdefault("flash_attn", False)
    extras = {k: kwargs.pop(k) for k in ("modality", "patch_size") if k in kwargs}
    import os
    import sys

    # Hub repos that import a top-level ``src`` package (AnySat does) break when the working
    # directory holds another ``src`` package, such as a GeoLattice checkout: hide it while loading.
    cwd = os.getcwd()
    shadow = os.path.isfile(os.path.join(cwd, "src", "__init__.py"))
    saved = list(sys.path)
    if shadow:
        sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != cwd]
        stale = {k: sys.modules.pop(k) for k in list(sys.modules)
                 if k == "src" or k.startswith("src.")}
    try:
        model = torch.hub.load(repo, entry, **kwargs)
    finally:
        sys.path[:] = saved
        if shadow:
            sys.modules.update({k: v for k, v in stale.items() if k not in sys.modules})
    return model, _fw_anysat if spec.name == "anysat" else _fw_plain, extras


def _load_transformers(spec, **kwargs):
    tr = _require("transformers")
    kwargs.pop("pretrained", None)
    model = tr.AutoModel.from_pretrained(spec.target, **kwargs)
    return model, _fw_hf


def load_model(name: str, *, pretrained: bool = True, device: Any = None,
               bands: Sequence[str] | None = None, wavelengths: Sequence[float] | None = None,
               pool: Literal["mean", "cls"] = "mean", **kwargs: Any) -> FoundationEncoder:
    """Build a foundation model and wrap it in a :class:`FoundationEncoder`.

    ``name`` is a catalog name or alias (see :func:`list_models` with ``loadable=True``).
    ``bands`` names the input bands in order (GeoLattice names such as ``"red"`` or Sentinel-2
    ids such as ``"B04"``); wavelength-aware models derive wavelengths from them, Prithvi
    selects matching patch embedding weights. Other keyword arguments go to the underlying
    builder, for example ``num_frames=3`` (Prithvi), ``modalities=["S2L2A", "S1GRD"]``
    (TerraMind), ``modality="sar"`` (CROMA) or ``patch_size=10`` (AnySat).

    Models whose loader is ``adapter`` need a local checkpoint: use the classes in
    :mod:`geolattice.rsense.dl.foundation`. AlphaEarth lives in Earth Engine: use
    :func:`alphaearth_image`. ``external`` models are listed for reference only.
    """
    spec = get_model_spec(name)
    if spec.loader == "earthengine":
        raise ValueError(f"{spec.name} is an Earth Engine dataset; use alphaearth_image() or "
                         "sample_alphaearth()")
    if spec.loader == "adapter":
        raise ValueError(f"{spec.name} needs a local checkpoint or extra package; use "
                         f"{spec.target}")
    if spec.loader == "external":
        raise NotImplementedError(f"{spec.name} has no GeoLattice loader yet; weights: {spec.weights}")
    dev = _resolve_device(device)
    extras: dict[str, Any] = {}
    if spec.loader == "torchgeo":
        model, forward, wavelengths, extras = _load_torchgeo(spec, bands, wavelengths,
                                                             pretrained=pretrained, **kwargs)
    elif spec.loader == "terratorch":
        model, forward, wavelengths, extras = _load_terratorch(spec, bands, wavelengths,
                                                               pretrained=pretrained, **kwargs)
    elif spec.loader == "open_clip":
        model, forward, extras = _load_open_clip(spec, **kwargs)
    elif spec.loader == "torch.hub":
        model, forward, extras = _load_hub(spec, pretrained=pretrained, **kwargs)
    else:
        model, forward = _load_transformers(spec, **kwargs)
    model = model.to(dev).eval()
    return FoundationEncoder(spec, model, device=dev, forward=forward, bands=bands,
                             wavelengths=wavelengths, pool=pool, extras=extras)


def segmentation_model(num_classes: int, *, backbone: str = "prithvi-eo-2.0-300m",
                       bands: Sequence[str] | None = None, decoder: str = "UperNetDecoder",
                       **kwargs: Any):
    """TerraTorch encoder-decoder for semantic segmentation on a catalog backbone.

    Works for TerraTorch-loaded models (Prithvi, TerraMind) and any TerraTorch backbone name.
    For catalog ViT backbones the necks that turn tokens from four evenly spaced blocks into
    image features are added unless ``necks=`` is given. For a Lightning training config use
    :func:`terratorch_config`.
    """
    fac = _require("terratorch.models")
    try:
        spec = get_model_spec(backbone)
        target = spec.target if spec.loader == "terratorch" else backbone
        default_bands = spec.defaults.get("bands")
    except KeyError:
        spec, target, default_bands = None, backbone, None
    args = {"backbone_bands": [b.upper() for b in (bands or default_bands)]} \
        if (bands or default_bands) else {}
    depth = {768: 12, 1024: 24, 1280: 32}.get(spec.embed_dim) if spec else None
    if spec is not None and spec.loader == "terratorch" and depth and "necks" not in kwargs:
        kwargs["necks"] = [{"name": "SelectIndices",
                            "indices": [depth // 4 * i - 1 for i in range(1, 5)]},
                           {"name": "ReshapeTokensToImage"}]
    return fac.EncoderDecoderFactory().build_model(
        task="segmentation", backbone=target.removeprefix("terratorch_"),
        backbone_pretrained=True, decoder=decoder, num_classes=num_classes, **args, **kwargs)


def zero_shot_classify(images: Sequence[Any], class_names: Sequence[str], *,
                       model: str = "remoteclip-vit-b-32", **kwargs: Any) -> pd.DataFrame:
    """Zero-shot scene classification with a vision-language model (RemoteCLIP, GeoRSCLIP)."""
    from .classify.image_classification import zero_shot_classify as _zs

    spec = get_model_spec(model)
    if spec.loader != "open_clip":
        raise ValueError(f"{spec.name} is not a CLIP-style model; choose one of "
                         f"{[m.name for m in _ENTRIES if m.loader == 'open_clip']}")
    return _zs(images, class_names, model=spec.target, **kwargs)


def terratorch_config(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """TerraTorch fine-tuning config; see
    :func:`geolattice.rsense.classify.image_classification.terratorch_config`."""
    from .classify.image_classification import terratorch_config as _cfg

    return _cfg(*args, **kwargs)


# =============================================================================================
# 4. AlphaEarth Foundations (Google Satellite Embedding, Earth Engine)
# =============================================================================================

ALPHAEARTH_COLLECTION = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
ALPHAEARTH_BANDS = [f"A{i:02d}" for i in range(64)]


_EE_READY = False


def _ee(project: str | None = None):
    """Earth Engine module, initialized once (with ``project`` when given)."""
    global _EE_READY
    ee = _require("ee", "rsense-gee")
    if project is not None:
        ee.Initialize(project=project)
        _EE_READY = True
    elif not _EE_READY:
        try:
            ee.Number(1).getInfo()
        except Exception:
            ee.Initialize()
        _EE_READY = True
    return ee


def _ee_geometry(ee, region):
    if region is None or isinstance(region, ee.Geometry) or isinstance(region, ee.FeatureCollection):
        return region
    if hasattr(region, "to_crs") and hasattr(region, "geometry"):  # GeoDataFrame / GeoSeries
        g = region.to_crs(4326)
        geom = g.geometry.union_all() if hasattr(g.geometry, "union_all") else g.unary_union
        return ee.Geometry(geom.__geo_interface__)
    if hasattr(region, "__geo_interface__"):
        return ee.Geometry(region.__geo_interface__)
    if isinstance(region, Mapping):
        return ee.Geometry(region.get("geometry", region))
    if isinstance(region, Sequence) and len(region) == 4:
        return ee.Geometry.Rectangle(list(region))
    raise TypeError("region must be a GeoDataFrame, shapely geometry, GeoJSON dict or bbox")


def alphaearth_image(region: Any = None, year: int = 2024, *, project: str | None = None):
    """Annual AlphaEarth embedding image (64 bands, A00 to A63) for ``year`` as an ``ee.Image``.

    Tiles overlapping ``region`` (GeoDataFrame, shapely geometry, GeoJSON or
    ``(xmin, ymin, xmax, ymax)`` in degrees) are mosaicked and clipped. The data starts in 2017.
    Embeddings are unit length, so a dot product between pixels is their cosine similarity.
    """
    ee = _ee(project)
    col = ee.ImageCollection(ALPHAEARTH_COLLECTION).filterDate(f"{year}-01-01",
                                                               f"{year + 1}-01-01")
    geom = _ee_geometry(ee, region)
    if geom is not None:
        col = col.filterBounds(geom)
    img = col.mosaic().select(ALPHAEARTH_BANDS)
    return img.clip(geom) if geom is not None else img


def _sample_alphaearth_ee(points: Any, year: int | Sequence[int] = 2024, *, scale: float = 10.0,
                      id_column: str | None = None, chunk: int = 4000,
                      project: str | None = None) -> pd.DataFrame:
    """AlphaEarth embeddings at point locations, columns A00 to A63.

    ``points`` is a GeoDataFrame of points (any CRS). Other columns are kept, so the result
    is ready for a regression or classification model. Points are sent in chunks of
    ``chunk`` to stay under Earth Engine request limits. With several ``year`` values the
    result is a long table with one row per point and year and a ``year`` column.
    """
    if not isinstance(year, (int, np.integer)):
        frames = [_sample_alphaearth_ee(points, int(y), scale=scale, id_column=id_column,
                                        chunk=chunk, project=project).assign(year=int(y))
                  for y in year]
        return pd.concat(frames, ignore_index=True)
    ee = _ee(project)
    gdf = points.to_crs(4326).reset_index(drop=True)
    key = id_column or "_pt"
    gdf = gdf.assign(**{key: gdf[id_column] if id_column else np.arange(len(gdf))})
    img = alphaearth_image(None, year, project=None)
    frames = []
    for s in range(0, len(gdf), chunk):
        part = gdf.iloc[s:s + chunk]
        feats = [ee.Feature(ee.Geometry.Point([float(p.x), float(p.y)]), {key: _py(v)})
                 for p, v in zip(part.geometry, part[key])]
        fc = img.sampleRegions(collection=ee.FeatureCollection(feats), properties=[key],
                               scale=scale, geometries=False)
        info = fc.getInfo()
        frames.append(pd.DataFrame([f["properties"] for f in info.get("features", [])]))
    emb = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=[key])
    out = gdf.merge(emb, on=key, how="left")
    if not id_column:
        out = out.drop(columns=key)
    return out


def _py(v):
    return v.item() if hasattr(v, "item") else v


def _alphaearth_to_xarray_ee(region: Any, year: int = 2024, *, scale: float = 10.0,
                         crs: str = "EPSG:4326", bands: Sequence[str] | None = None,
                         project: str | None = None, max_pixels: int = 25_000_000,
                         tile: int = 256) -> xr.DataArray:
    """Download AlphaEarth embeddings for a region as a ``(embedding, y, x)`` DataArray.

    Uses ``ee.data.computePixels`` in tiles of ``tile`` x ``tile`` pixels (Earth Engine limits
    each response to about 48 MB; 256 x 256 x 64 float32 values are 16 MB) and mosaics them.
    Suits areas up to a few thousand pixels on a side; for larger areas export to Drive or Cloud
    Storage. ``scale`` is in metres (converted to degrees for geographic CRSs); with a ``crs`` in
    metres, such as a UTM zone, the grid is square in metres. ``bands`` picks a subset of A00 to
    A63. Coarser scales than 10 m take the nearest pixel, as Earth Engine does by default.
    """
    ee = _ee(project)
    bands = list(bands or ALPHAEARTH_BANDS)
    geom = _ee_geometry(ee, region)
    xmin, ymin, xmax, ymax = _bounds(geom.bounds(proj=crs, maxError=1).getInfo())
    step = scale / 111_320.0 if crs.upper() in ("EPSG:4326", "EPSG:4269") else scale
    nx, ny = max(int(np.ceil((xmax - xmin) / step)), 1), max(int(np.ceil((ymax - ymin) / step)), 1)
    if nx * ny > max_pixels:
        raise ValueError(f"{nx}x{ny} pixels is too large for computePixels; increase scale, "
                         "shrink the region or export the image")
    img = alphaearth_image(region, year).select(bands)
    data = np.full((len(bands), ny, nx), np.nan, dtype="float32")
    for r0 in range(0, ny, tile):
        for c0 in range(0, nx, tile):
            h, w = min(tile, ny - r0), min(tile, nx - c0)
            arr = ee.data.computePixels({
                "expression": img, "fileFormat": "NUMPY_NDARRAY",
                "grid": {"dimensions": {"width": w, "height": h},
                         "affineTransform": {"scaleX": step, "shearX": 0,
                                             "translateX": xmin + c0 * step, "shearY": 0,
                                             "scaleY": -step, "translateY": ymax - r0 * step},
                         "crsCode": crs}})
            data[:, r0:r0 + h, c0:c0 + w] = np.stack([np.asarray(arr[b], dtype="float32")
                                                       for b in bands])
    xs = xmin + step * (np.arange(nx) + 0.5)
    ys = ymax - step * (np.arange(ny) + 0.5)
    da = xr.DataArray(data, dims=("embedding", "y", "x"),
                      coords={"embedding": bands, "y": ys, "x": xs}, name="alphaearth",
                      attrs={"year": year, "scale": scale, "source": ALPHAEARTH_COLLECTION})
    try:
        da = da.rio.write_crs(crs)
    except Exception:
        pass
    return da


# ---- AlphaEarth without Earth Engine: public copies of the embeddings ------------------------

#: Source Cooperative GeoZarr mosaic of the annual embeddings (EPSG:4326, about 10 m, int8),
#: dimensions ``(time, band, y, x)`` with years 2017 onward; read anonymously over HTTPS.
ALPHAEARTH_ZARR = "https://data.source.coop/tge-labs/aef-mosaic/"
#: Google Cloud Storage copy: 8192 x 8192 Cloud Optimized GeoTIFFs in UTM zones, with overviews.
ALPHAEARTH_GCS = "alphaearth_foundations/satellite_embedding/v1/annual"
_AEF_GSD = 0.00008983111749910169  # degrees per pixel of the Zarr mosaic
_AEF_X0, _AEF_Y0 = -180.0, 83.68570533713473  # upper-left corner of the mosaic
_AEF_FIRST_YEAR = 2017
_AEF_ARRAY = None


def alphaearth_dequantize(values: Any) -> np.ndarray:
    """Convert the int8 values of the public AlphaEarth files to float embeddings.

    The files store ``sign(v) * sqrt(|v|) * 127.5``; this inverts it,
    ``(v / 127.5) ** 2 * sign(v)``, and turns the no-data value -128 into NaN.
    """
    v = np.asarray(values)
    out = ((v.astype("float32") / 127.5) ** 2) * np.sign(v).astype("float32")
    return np.where(v == -128, np.nan, out).astype("float32")


def _aef_array():
    global _AEF_ARRAY
    if _AEF_ARRAY is None:
        zarr = _require("zarr", "rsense-foundation")
        store = zarr.storage.FsspecStore.from_url(ALPHAEARTH_ZARR, read_only=True)
        _AEF_ARRAY = zarr.open_array(store, path="embeddings", mode="r")
    return _AEF_ARRAY


def _aef_time(year: int) -> int:
    t = int(year) - _AEF_FIRST_YEAR
    if not 0 <= t < _aef_array().shape[0]:
        raise ValueError(f"AlphaEarth years are {_AEF_FIRST_YEAR} to "
                         f"{_AEF_FIRST_YEAR + _aef_array().shape[0] - 1}")
    return t


def _wgs84_bounds(region: Any) -> tuple[float, float, float, float]:
    if hasattr(region, "to_crs") and hasattr(region, "total_bounds"):
        return tuple(region.to_crs(4326).total_bounds)
    if hasattr(region, "rio"):
        return tuple(region.rio.transform_bounds("EPSG:4326"))
    if hasattr(region, "bounds") and not isinstance(region, (tuple, list)):
        return tuple(region.bounds)
    if isinstance(region, Mapping):
        from shapely.geometry import shape

        return tuple(shape(region.get("geometry", region)).bounds)
    if isinstance(region, Sequence) and len(region) == 4:
        return tuple(float(v) for v in region)
    raise TypeError("region must be a GeoDataFrame, DataArray, shapely geometry, GeoJSON or bbox")


def _alphaearth_zarr_window(bounds: tuple[float, float, float, float], year: int,
                            bands: Sequence[str]) -> xr.DataArray:
    """Embeddings in a lon/lat box at the mosaic's native resolution, as float32."""
    a = _aef_array()
    xmin, ymin, xmax, ymax = bounds
    r0, r1 = int((_AEF_Y0 - ymax) / _AEF_GSD), int(np.ceil((_AEF_Y0 - ymin) / _AEF_GSD))
    c0, c1 = int((xmin - _AEF_X0) / _AEF_GSD), int(np.ceil((xmax - _AEF_X0) / _AEF_GSD))
    bi = [ALPHAEARTH_BANDS.index(b) for b in bands]
    raw = a[_aef_time(year), :, r0:r1, c0:c1]
    data = alphaearth_dequantize(raw[bi])
    xs = _AEF_X0 + _AEF_GSD * (np.arange(c0, c1) + 0.5)
    ys = _AEF_Y0 - _AEF_GSD * (np.arange(r0, r1) + 0.5)
    return xr.DataArray(data, dims=("embedding", "y", "x"),
                        coords={"embedding": list(bands), "y": ys, "x": xs}).rio.write_crs(4326)


def alphaearth_index(year: int | None = None) -> pd.DataFrame:
    """Index of the AlphaEarth GeoTIFF tiles on Google Cloud Storage (path, year, UTM zone, bounds).

    Read anonymously once and cached in ``~/.cache/geolattice/aef_index.parquet``.
    """
    from pathlib import Path

    cache = Path.home() / ".cache" / "geolattice" / "aef_index.parquet"
    if cache.exists():
        idx = pd.read_parquet(cache)
    else:
        gcsfs = _require("gcsfs", "web-cloud")
        fs = gcsfs.GCSFileSystem(token="anon")
        with fs.open(f"{ALPHAEARTH_GCS}/aef_index.csv") as f:
            idx = pd.read_csv(f)
        cache.parent.mkdir(parents=True, exist_ok=True)
        idx.to_parquet(cache)
    return idx if year is None else idx[idx["year"] == int(year)].reset_index(drop=True)


def _alphaearth_gcs_grid(bounds_wgs84, year, crs, scale, bands, dst_transform, dst_shape):
    """Mosaic GeoTIFF tiles (read from overviews near ``scale``) onto a target grid."""
    import os

    import rasterio
    from rasterio.warp import Resampling, reproject

    os.environ.setdefault("GS_NO_SIGN_REQUEST", "YES")
    idx = alphaearth_index(year)
    xmin, ymin, xmax, ymax = bounds_wgs84
    hits = idx[(idx.wgs84_east >= xmin) & (idx.wgs84_west <= xmax) &
               (idx.wgs84_north >= ymin) & (idx.wgs84_south <= ymax)]
    bi = [ALPHAEARTH_BANDS.index(b) + 1 for b in bands]
    out = np.full((len(bands),) + tuple(dst_shape), np.nan, "float32")
    for path in hits["path"]:
        with rasterio.open(path.replace("gs://", "/vsigs/")) as src:
            f = max(1, int(scale // abs(src.res[0])))
            h, w = max(1, src.height // f), max(1, src.width // f)
            raw = src.read(bi, out_shape=(len(bi), h, w))  # overviews do the decimation
            tr = src.transform * src.transform.scale(src.width / w, src.height / h)
            part = np.full_like(out, np.nan)
            reproject(alphaearth_dequantize(raw), part, src_transform=tr, src_crs=src.crs,
                      src_nodata=np.nan, dst_transform=dst_transform, dst_crs=crs,
                      dst_nodata=np.nan, resampling=Resampling.nearest)
            fill = np.isnan(out) & np.isfinite(part)
            out[fill] = part[fill]
    return out


def alphaearth_to_xarray(region: Any, year: int = 2024, *, scale: float = 10.0,
                         crs: str = "EPSG:4326", bands: Sequence[str] | None = None,
                         source: Literal["auto", "zarr", "gcs", "earthengine"] = "auto",
                         like: xr.DataArray | None = None, project: str | None = None,
                         **kwargs: Any) -> xr.DataArray:
    """AlphaEarth embeddings for a region as a ``(embedding, y, x)`` DataArray.

    ``source`` picks where they come from, none of which needs an account except Earth Engine:

    - ``"zarr"``: the Source Cooperative GeoZarr mosaic, read at its native ~10 m resolution,
      then resampled to ``crs`` and ``scale``. Best for areas up to tens of kilometres.
    - ``"gcs"``: the Google Cloud Storage GeoTIFFs, read from overviews close to ``scale``.
      Best for coarse maps of large regions.
    - ``"earthengine"``: ``ee.data.computePixels`` (needs ``earthengine authenticate`` and a
      registered ``project``).
    - ``"auto"``: ``zarr`` for ``scale`` below 100 m, otherwise ``gcs``.

    With ``like`` (a georeferenced DataArray) the result is put on exactly that grid. ``scale``
    is in metres (converted to degrees for geographic CRSs). The values are float embeddings of
    length close to 1; no-data is NaN.
    """
    bands = list(bands or ALPHAEARTH_BANDS)
    if source == "auto":
        source = "zarr" if scale < 100 else "gcs"
    if source == "earthengine":
        da = _alphaearth_to_xarray_ee(region, year, scale=scale, crs=crs, bands=bands,
                                      project=project, **kwargs)
        return da.rio.reproject_match(like) if like is not None else da
    from rasterio.transform import from_origin
    from rasterio.warp import transform_bounds

    if like is not None:
        crs = like.rio.crs
        dst_transform, dst_shape = like.rio.transform(), (like.sizes["y"], like.sizes["x"])
        bounds = like.rio.transform_bounds("EPSG:4326")
    else:
        bounds = _wgs84_bounds(region)
        geographic = str(crs).upper() in ("EPSG:4326", "EPSG:4269")
        step = scale / 111_320.0 if geographic else scale
        xmin, ymin, xmax, ymax = bounds if geographic else transform_bounds(4326, crs, *bounds)
        nx, ny = max(int(np.ceil((xmax - xmin) / step)), 1), max(int(np.ceil((ymax - ymin) / step)), 1)
        dst_transform, dst_shape = from_origin(xmin, ymax, step, step), (ny, nx)
    if source == "zarr":
        pad = 2 * _AEF_GSD
        win = _alphaearth_zarr_window((bounds[0] - pad, bounds[1] - pad, bounds[2] + pad,
                                       bounds[3] + pad), year, bands)
        from rasterio.enums import Resampling

        coarse = abs(dst_transform.a) > 3 * (_AEF_GSD if str(crs).upper() == "EPSG:4326" else 10.0)
        da = win.rio.reproject(crs, transform=dst_transform, shape=dst_shape,
                               resampling=Resampling.average if coarse else Resampling.nearest,
                               nodata=np.nan)
        data = da.values.astype("float32")
    elif source == "gcs":
        data = _alphaearth_gcs_grid(bounds, year, crs, abs(dst_transform.a) if str(crs).upper()
                                    not in ("EPSG:4326",) else scale, bands, dst_transform,
                                    dst_shape)
    else:
        raise ValueError("source must be 'auto', 'zarr', 'gcs' or 'earthengine'")
    xs = dst_transform.c + dst_transform.a * (np.arange(dst_shape[1]) + 0.5)
    ys = dst_transform.f + dst_transform.e * (np.arange(dst_shape[0]) + 0.5)
    out = xr.DataArray(data, dims=("embedding", "y", "x"),
                       coords={"embedding": bands, "y": ys, "x": xs}, name="alphaearth",
                       attrs={"year": int(year), "scale": float(abs(dst_transform.a)),
                              "source": {"zarr": ALPHAEARTH_ZARR, "gcs": "gs://" + ALPHAEARTH_GCS
                                         }[source]})
    return out.rio.write_crs(crs)


def sample_alphaearth(points: Any, year: int | Sequence[int] = 2024, *,
                      source: Literal["zarr", "earthengine"] = "zarr", workers: int = 16,
                      id_column: str | None = None, project: str | None = None,
                      **kwargs: Any) -> pd.DataFrame:
    """AlphaEarth embeddings at point locations, columns A00 to A63 (one 10 m pixel each).

    ``points`` is a GeoDataFrame of points (any CRS); other columns are kept, so the result is
    ready for a regression or classification model. ``source="zarr"`` (default) reads the
    Source Cooperative mosaic anonymously, one small chunk per point, in ``workers`` parallel
    requests; ``source="earthengine"`` uses Earth Engine (needs a registered ``project``). With
    several ``year`` values the result is a long table with a ``year`` column.
    """
    if source == "earthengine":
        return _sample_alphaearth_ee(points, year, id_column=id_column, project=project, **kwargs)
    if source != "zarr":
        raise ValueError("source must be 'zarr' or 'earthengine'")
    if not isinstance(year, (int, np.integer)):
        return pd.concat([sample_alphaearth(points, int(y), source=source, workers=workers,
                                            id_column=id_column).assign(year=int(y))
                          for y in year], ignore_index=True)
    from concurrent.futures import ThreadPoolExecutor

    a = _aef_array()
    t = _aef_time(year)
    g = points.to_crs(4326).reset_index(drop=True)
    rows = ((_AEF_Y0 - g.geometry.y.to_numpy()) / _AEF_GSD).astype(int)
    cols = ((g.geometry.x.to_numpy() - _AEF_X0) / _AEF_GSD).astype(int)
    with ThreadPoolExecutor(max(1, workers)) as ex:
        vals = list(ex.map(lambda rc: np.asarray(a[t, :, rc[0], rc[1]]), zip(rows, cols)))
    emb = pd.DataFrame(alphaearth_dequantize(np.stack(vals)), columns=ALPHAEARTH_BANDS)
    return pd.concat([points.reset_index(drop=True), emb], axis=1)


def _bounds(geojson: Mapping[str, Any]) -> tuple[float, float, float, float]:
    ring = np.asarray(geojson["coordinates"][0], dtype="float64")
    return ring[:, 0].min(), ring[:, 1].min(), ring[:, 0].max(), ring[:, 1].max()


# =============================================================================================
# 5. Using embeddings: probes, similarity, clustering, visualization
# =============================================================================================


def embeddings_to_frame(emb: xr.DataArray, *, dropna: bool = True) -> pd.DataFrame:
    """Flatten an ``(embedding, y, x)`` map to a table with ``y``, ``x`` and one column per dim."""
    dim = emb.dims[0]
    a = emb.transpose(dim, "y", "x").values
    n = a.shape[0]
    flat = a.reshape(n, -1).T
    yy, xx = np.meshgrid(emb["y"].values, emb["x"].values, indexing="ij")
    cols = [str(c) if not np.issubdtype(type(c), np.integer) else f"e{int(c):03d}"
            for c in emb[dim].values]
    df = pd.DataFrame(flat, columns=cols)
    df.insert(0, "x", xx.ravel())
    df.insert(0, "y", yy.ravel())
    return df.dropna() if dropna else df


def similarity_map(emb: xr.DataArray, query: Any, *,
                   metric: Literal["cosine", "euclidean"] = "cosine") -> xr.DataArray:
    """Similarity of every cell of an embedding map to ``query``.

    ``query`` is a vector ``(D,)``, several vectors ``(k, D)`` (their mean is used) or a
    DataArray slice of ``emb``. Cosine returns similarity (1 = identical direction);
    euclidean returns distance (0 = identical).
    """
    dim = emb.dims[0]
    q = np.asarray(query.values if isinstance(query, xr.DataArray) else query, "float64")
    q = q.reshape(-1, emb.sizes[dim]).mean(0)
    a = emb.transpose(dim, ...).values.astype("float64")
    if metric == "cosine":
        num = np.tensordot(q, a, axes=(0, 0))
        den = np.linalg.norm(a, axis=0) * max(np.linalg.norm(q), 1e-12)
        out = num / np.maximum(den, 1e-12)
    elif metric == "euclidean":
        out = np.sqrt(((a - q.reshape(-1, *[1] * (a.ndim - 1))) ** 2).sum(0))
    else:
        raise ValueError("metric must be 'cosine' or 'euclidean'")
    res = xr.DataArray(out.astype("float32"), dims=emb.dims[1:],
                       coords={d: emb[d] for d in emb.dims[1:]}, name=f"{metric}_similarity")
    try:
        if emb.rio.crs is not None:
            res = res.rio.write_crs(emb.rio.crs)
    except Exception:
        pass
    return res


def cluster_embeddings(emb: xr.DataArray | np.ndarray, n_clusters: int = 8, *,
                       method: Literal["kmeans", "gmm"] = "kmeans", seed: int = 0):
    """Unsupervised map from embeddings: k-means or Gaussian mixture labels.

    For an ``(embedding, y, x)`` DataArray returns a ``(y, x)`` label map (NaN cells get -1);
    for an array ``(n, D)`` returns labels ``(n,)``.
    """
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture

    is_map = isinstance(emb, xr.DataArray)
    if is_map:
        dim = emb.dims[0]
        a = emb.transpose(dim, "y", "x").values
        X = a.reshape(a.shape[0], -1).T
    else:
        X = np.asarray(emb)
    ok = np.isfinite(X).all(1)
    est = KMeans(n_clusters, n_init=10, random_state=seed) if method == "kmeans" \
        else GaussianMixture(n_clusters, random_state=seed)
    labels = np.full(len(X), -1, dtype="int32")
    labels[ok] = est.fit_predict(X[ok])
    if not is_map:
        return labels
    return xr.DataArray(labels.reshape(emb.sizes["y"], emb.sizes["x"]), dims=("y", "x"),
                        coords={"y": emb["y"], "x": emb["x"]}, name="cluster")


def embedding_rgb(emb: xr.DataArray, *, method: Literal["pca", "first3"] = "pca",
                  percentile: tuple[float, float] = (2, 98)) -> xr.DataArray:
    """Three-band ``(band, y, x)`` image in [0, 1] for viewing an embedding map.

    ``pca`` projects onto the first three principal components (stretched by percentile);
    ``first3`` uses the first three embedding dimensions.
    """
    dim = emb.dims[0]
    a = emb.transpose(dim, "y", "x").values.astype("float64")
    D, ny, nx = a.shape
    X = a.reshape(D, -1).T
    ok = np.isfinite(X).all(1)
    Y = np.full((len(X), 3), np.nan)
    if method == "pca":
        Xc = X[ok] - X[ok].mean(0)
        _, _, vt = np.linalg.svd(Xc, full_matrices=False)
        Y[ok] = Xc @ vt[:3].T
    else:
        Y[ok] = X[ok, :3]
    lo, hi = np.nanpercentile(Y, percentile[0], 0), np.nanpercentile(Y, percentile[1], 0)
    Y = np.clip((Y - lo) / np.maximum(hi - lo, 1e-12), 0, 1)
    return xr.DataArray(Y.T.reshape(3, ny, nx).astype("float32"), dims=("band", "y", "x"),
                        coords={"band": ["r", "g", "b"], "y": emb["y"], "x": emb["x"]},
                        name="embedding_rgb")


def linear_probe(X: Any, y: Any, *, task: Literal["auto", "classification", "regression"] = "auto",
                 cv: int = 5, groups: Any = None, alpha: float = 1.0, seed: int = 0
                 ) -> dict[str, Any]:
    """Cross-validated linear probe: how much the embeddings know about ``y``.

    Logistic regression for classes, ridge regression for continuous targets, both on
    standardized features. Pass ``groups`` (for example spatial blocks) for spatial
    cross-validation; without them random folds overstate accuracy on autocorrelated data.
    Returns the fold scores, their mean and an estimator refit on all data.
    """
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, cross_validate
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X = np.asarray(X, "float64")
    y = np.asarray(y)
    if task == "auto":
        task = "classification" if (y.dtype.kind in "OUSb" or
                                    (y.dtype.kind in "iu" and len(np.unique(y)) <= 50)) \
            else "regression"
    if task == "classification":
        est = make_pipeline(StandardScaler(), LogisticRegression(C=1.0 / alpha, max_iter=2000))
        scoring = {"accuracy": "accuracy", "f1_macro": "f1_macro"}
        splitter = StratifiedKFold(cv, shuffle=True, random_state=seed)
    else:
        est = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        scoring = {"r2": "r2", "rmse": "neg_root_mean_squared_error"}
        splitter = KFold(cv, shuffle=True, random_state=seed)
    if groups is not None:
        splitter = GroupKFold(cv)
    res = cross_validate(est, X, y, cv=splitter, groups=groups, scoring=scoring)
    scores = {k.removeprefix("test_"): (-v if k == "test_rmse" else v)
              for k, v in res.items() if k.startswith("test_")}
    return {"task": task, "scores": scores, "mean": {k: float(np.mean(v)) for k, v in scores.items()},
            "estimator": est.fit(X, y), "cv": "group" if groups is not None else "random"}


def knn_probe(X: Any, y: Any, *, k: int = 5, cv: int = 5, groups: Any = None,
              metric: str = "cosine", seed: int = 0) -> dict[str, Any]:
    """Cross-validated k-nearest-neighbour probe (accuracy for classes, R² otherwise)."""
    from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, cross_val_score
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

    X = np.asarray(X, "float64")
    y = np.asarray(y)
    cls = y.dtype.kind in "OUSb" or (y.dtype.kind in "iu" and len(np.unique(y)) <= 50)
    est = (KNeighborsClassifier if cls else KNeighborsRegressor)(n_neighbors=k, metric=metric)
    splitter = GroupKFold(cv) if groups is not None else (
        StratifiedKFold(cv, shuffle=True, random_state=seed) if cls
        else KFold(cv, shuffle=True, random_state=seed))
    s = cross_val_score(est, X, y, cv=splitter, groups=groups)
    return {"task": "classification" if cls else "regression", "scores": s,
            "mean": float(s.mean()), "metric": "accuracy" if cls else "r2"}


def embedding_search(index: Any, query: Any, *, k: int = 10,
                     metric: Literal["cosine", "euclidean"] = "cosine"):
    """Nearest neighbours of ``query`` rows in ``index`` → ``(indices, scores)``."""
    from .classify.image_classification import embedding_search as _search

    return _search(np.asarray(index), np.asarray(query), k=k, metric=metric)


def benchmark_encoders(inputs: Mapping[str, Mapping[str, Any]], y: Any, *, groups: Any = None,
                       cv: int = 5, device: Any = None, batch_size: int = 32,
                       probes: Sequence[str] = ("linear", "knn"), k: int = 5,
                       seed: int = 0) -> pd.DataFrame:
    """Compare encoders on one labelled set of chips with linear and kNN probes.

    ``inputs`` maps a row label to a dict with either ``"embeddings"`` (a precomputed
    ``(n, D)`` array, for baselines such as band means) or ``"x"`` (chips prepared for that
    model) plus optional ``"model"`` (catalog name, default the row label), ``"load"``
    (keyword arguments for :func:`load_model`) and ``"encode"`` (keyword arguments for
    :meth:`FoundationEncoder.embed`). ``groups`` turns on spatial cross-validation.

    Returns one row per input with the embedding size, parameters (millions), seconds per
    1000 chips, peak GPU memory (MB, when on CUDA) and the mean probe scores.
    """
    import time as _time

    rows = []
    for label, spec in inputs.items():
        row: dict[str, Any] = {"input": label}
        if "embeddings" in spec:
            E = np.asarray(spec["embeddings"], "float32")
            row.update(params_m=0.0, s_per_1000=np.nan, gpu_mb=np.nan)
        else:
            torch = _torch()
            enc = load_model(spec.get("model", label), device=device, **spec.get("load", {}))
            dev = enc.device
            cuda = getattr(dev, "type", str(dev)) == "cuda"
            if cuda:
                torch.cuda.synchronize(dev)
                torch.cuda.reset_peak_memory_stats(dev)
            t0 = _time.perf_counter()
            E = enc.embed(spec["x"], batch_size=batch_size, **spec.get("encode", {}))
            if cuda:
                torch.cuda.synchronize(dev)
            dt = _time.perf_counter() - t0
            row.update(params_m=sum(p.numel() for p in enc.model.parameters()) / 1e6,
                       s_per_1000=1000 * dt / max(len(E), 1),
                       gpu_mb=torch.cuda.max_memory_allocated(dev) / 2**20 if cuda else np.nan)
            del enc
            if cuda:
                torch.cuda.empty_cache()
        row["embed_dim"] = E.shape[1]
        if "linear" in probes:
            lp = linear_probe(E, y, cv=cv, groups=groups, seed=seed)
            row.update({f"linear_{m}": v for m, v in lp["mean"].items()})
        if "knn" in probes:
            kp = knn_probe(E, y, k=k, cv=cv, groups=groups, seed=seed)
            row[f"knn_{kp['metric']}"] = kp["mean"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("input")


def embedding_change(emb_a: xr.DataArray, emb_b: xr.DataArray) -> xr.DataArray:
    """Per-cell change between two embedding maps of the same grid: 1 - cosine similarity.

    0 means no change in what the model sees; values near 1 or above mean the surface
    changed. Works for AlphaEarth years or any two :meth:`FoundationEncoder.embed_raster`
    outputs.
    """
    dim = emb_a.dims[0]
    a = emb_a.transpose(dim, ...).values.astype("float64")
    b = emb_b.transpose(emb_b.dims[0], ...).values.astype("float64")
    if a.shape != b.shape:
        raise ValueError(f"embedding maps differ in shape: {a.shape} vs {b.shape}")
    cos = (a * b).sum(0) / np.maximum(np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0), 1e-12)
    out = xr.DataArray((1 - cos).astype("float32"), dims=emb_a.dims[1:],
                       coords={d: emb_a[d] for d in emb_a.dims[1:]}, name="embedding_change")
    try:
        if emb_a.rio.crs is not None:
            out = out.rio.write_crs(emb_a.rio.crs)
    except Exception:
        pass
    return out


__all__ = [
    "ALIASES", "ALPHAEARTH_BANDS", "ALPHAEARTH_COLLECTION", "BANDWIDTHS_NM", "CATEGORIES",
    "LOADERS", "MODELS", "SAR_CHANNEL_IDS", "WAVELENGTHS_NM", "FoundationEncoder",
    "FoundationModel", "alphaearth_image", "alphaearth_to_xarray", "band_wavelengths", "INPUT_SPECS", "input_spec", "prepare",
    "ALPHAEARTH_ZARR", "ALPHAEARTH_GCS", "alphaearth_dequantize", "alphaearth_index",
    "benchmark_encoders", "chip_metadata", "chip_raster", "cluster_embeddings",
    "embedding_change", "embedding_rgb", "embedding_search",
    "embeddings_to_frame", "get_model_spec", "knn_probe", "linear_probe", "list_models",
    "load_model", "model_info", "normalize", "pool_features", "recommend", "sample_alphaearth",
    "segmentation_model", "similarity_map", "terratorch_config", "zero_shot_classify",
]
