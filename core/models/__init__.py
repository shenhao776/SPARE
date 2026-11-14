import os
import sys
from types import SimpleNamespace

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as tvf
import einops

from core.config import ROOT_DIR, vpr_logger

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def _make_torch_hub_cache_first():

    hub = torch.hub
    if getattr(hub, "_star_vpr_cache_first", False):
        return
    original = hub._parse_repo_info

    def _parse_repo_info(github):
        if ":" not in github and "/" in github:
            owner, _, name = github.partition("/")
            for ref in ("main", "master"):
                if os.path.isdir(os.path.join(hub.get_dir(), f"{owner}_{name}_{ref}")):
                    return owner, name, ref
        return original(github)

    hub._parse_repo_info = _parse_repo_info
    hub._star_vpr_cache_first = True


_make_torch_hub_cache_first()

_REGISTRY = {}


def register(name):
    def deco(fn):
        _REGISTRY[name] = fn
        return fn

    return deco


def build_model(model_cfg, device):

    name = model_cfg["name"]
    if name not in _REGISTRY:
        raise ValueError(
            f"Unsupported VPR model '{name}'. Registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name](model_cfg, device)


def available_models():
    return sorted(_REGISTRY)


def _abspath(rel):
    return rel if os.path.isabs(rel) else os.path.join(ROOT_DIR, rel)


class ResizeToMultipleOf14:
    def __init__(self, target_short_edge=322):
        self.target_short_edge = target_short_edge

    def __call__(self, img):
        w, h = img.size
        if h < w:
            new_h = self.target_short_edge
            new_w = int(new_h * (w / h))
        else:
            new_w = self.target_short_edge
            new_h = int(new_w * (h / w))

        new_w = round(new_w / 14.0) * 14
        new_h = round(new_h / 14.0) * 14

        from PIL import Image

        return img.resize((new_w, new_h), Image.Resampling.BILINEAR)


def get_transform(model_cfg):

    tf = model_cfg.get("transform", {}) or {}
    ttype = tf.get("type", "multiple_of_14")

    if ttype == "max_edge_plain":
        max_edge = int(tf.get("max_edge", 640))

        class _MaxEdge:
            def __call__(self, img):
                from PIL import Image

                w, h = img.size
                s = max_edge / max(w, h)
                if s < 1.0:
                    img = img.resize(
                        (round(w * s), round(h * s)), Image.Resampling.BILINEAR
                    )
                return img

        return transforms.Compose([_MaxEdge(), transforms.ToTensor()])

    if ttype == "resize_square":
        size = int(tf["size"])
        if tf.get("interp") == "bicubic":
            return transforms.Compose(
                [
                    transforms.Resize(
                        (size, size),
                        interpolation=transforms.InterpolationMode.BICUBIC,
                        antialias=True,
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
                ]
            )
        return transforms.Compose(
            [
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ]
        )

    short_edge = int(tf.get("short_edge", 322))
    return transforms.Compose(
        [
            ResizeToMultipleOf14(short_edge),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]
    )


def _build_torch_hub(cfg, device):
    hub_kwargs = cfg.get("hub_kwargs") or {}
    return (
        torch.hub.load(cfg["hub_repo"], cfg["hub_entry"], **hub_kwargs)
        .to(device)
        .eval()
    )


register("MegaLoc")(_build_torch_hub)
register("SALAD")(_build_torch_hub)
register("EigenPlaces")(_build_torch_hub)


@register("MixVPR")
def _build_mixvpr(cfg, device):
    mix_vpr_path = _abspath(cfg.get("third_party", "3rdparty/MixVPR"))
    if mix_vpr_path not in sys.path:
        sys.path.append(mix_vpr_path)
    try:
        from models.helper import get_backbone, get_aggregator
    except ImportError as e:
        raise ImportError(
            "❌ Failed to import MixVPR. Please ensure it exists under 3rdparty/MixVPR."
        ) from e

    class MixVPRWrapper(torch.nn.Module):
        def __init__(self, device):
            super().__init__()
            self.device = device
            self.backbone = get_backbone(
                backbone_arch="resnet50",
                pretrained=False,
                layers_to_freeze=2,
                layers_to_crop=[4],
            )
            agg_config = {
                "in_channels": 1024,
                "in_h": 20,
                "in_w": 20,
                "out_channels": 1024,
                "mix_depth": 4,
                "mlp_ratio": 1,
                "out_rows": 4,
            }
            self.aggregator = get_aggregator("mixvpr", agg_config)

        def forward(self, img_pt):
            _, _, h, w = img_pt.shape
            if h != 320 or w != 320:
                img_pt = tvf.resize(img_pt, [320, 320], antialias=True)
            x = self.backbone(img_pt)
            desc = self.aggregator(x)
            return desc.to(self.device)

    model = MixVPRWrapper(device)
    ckpt_path = _abspath(cfg["checkpoint"])
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            "❌ [MixVPR] Missing checkpoint file: "
            f"{ckpt_path}. Please place resnet50_MixVPR_4096.ckpt there."
        )
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("model.", "") if k.startswith("model.") else k
        cleaned_state_dict[new_key] = v
    model.load_state_dict(cleaned_state_dict, strict=False)
    vpr_logger.info("✅ [MixVPR] Loaded 4096-dim pretrained weights.")
    return model.to(device).eval()


@register("SuperVLAD")
def _build_supervlad(cfg, device):
    super_vlad_path = _abspath(cfg.get("third_party", "3rdparty/SuperVLAD"))
    if super_vlad_path not in sys.path:
        sys.path.append(super_vlad_path)
    try:
        from model.network import SuperVLADModel
    except ImportError as e:
        raise ImportError(
            "❌ Failed to import SuperVLAD. Please ensure 3rdparty/SuperVLAD is available."
        ) from e

    supervlad_args = SimpleNamespace(
        backbone=cfg.get("backbone", "dino"),
        crossimage_encoder=cfg.get("cross_image_encoder", True),
        supervlad_clusters=cfg.get("clusters", 4),
        ghost_clusters=cfg.get("ghost_clusters", 1),
        freeze_te=cfg.get("freeze_te", 8),
        fc_output_dim=None,
        device=device,
        trunc_te=None,
        pretrain="imagenet",
        aggregation="gem",
    )

    foundation_model_path = _abspath(cfg["foundation_model_path"])
    pretrained_foundation = os.path.exists(foundation_model_path)
    if not pretrained_foundation:
        raise FileNotFoundError(
            "❌ [SuperVLAD] Missing foundation .pth file: "
            f"{foundation_model_path}. Please download dinov2_vitb14_pretrain.pth."
        )

    model = SuperVLADModel(
        supervlad_args,
        pretrained_foundation=pretrained_foundation,
        foundation_model_path=foundation_model_path,
    )

    ckpt_path = _abspath(cfg["checkpoint"])
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            "❌ [SuperVLAD] Missing checkpoint .pth file: "
            f"{ckpt_path}. Please download official SuperVLAD.pth."
        )
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )
    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    vpr_logger.info(f"✅ [SuperVLAD] Loaded checkpoint: {ckpt_path}")

    expected_dim = int(cfg.get("clusters", 4)) * 768
    if int(cfg["feature_dim"]) != expected_dim:
        vpr_logger.warning(
            f"⚠️ [SuperVLAD] Config dim mismatch: feature_dim={cfg['feature_dim']}, "
            f"expected={expected_dim}. Please align clusters in the model YAML."
        )
    return model.to(device).eval()


@register("BoQ")
def _build_boq(cfg, device):
    boq_path = _abspath(cfg.get("third_party", "3rdparty/Bag-of-Queries"))
    if boq_path not in sys.path:
        sys.path.append(boq_path)

    boq_backbone = cfg.get("backbone", "dinov2")
    boq_output_dim = int(cfg.get("output_dim", cfg["feature_dim"]))

    try:
        boq_model = torch.hub.load(
            boq_path,
            "get_trained_boq",
            backbone_name=boq_backbone,
            output_dim=boq_output_dim,
            source="local",
        )
    except Exception as e:
        raise RuntimeError(
            f"❌ Failed to load BoQ backend (backbone={boq_backbone}, output_dim={boq_output_dim})."
        ) from e

    class BoQWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, img_pt):
            out = self.model(img_pt)
            if isinstance(out, (list, tuple)):
                return out[0]
            return out

    vpr_logger.info(
        f"✅ [BoQ] Loaded backbone={boq_backbone}, output_dim={boq_output_dim}."
    )
    return BoQWrapper(boq_model).to(device).eval()


@register("CricaVPR")
def _build_cricavpr(cfg, device):
    crica_path = _abspath(cfg.get("third_party", "3rdparty/CricaVPR"))
    if crica_path not in sys.path:
        sys.path.append(crica_path)

    try:
        from network import CricaVPRNet
    except ImportError as e:
        raise ImportError(
            "❌ Failed to import CricaVPR network. Please ensure 3rdparty/CricaVPR is available."
        ) from e

    foundation_model_path = _abspath(cfg["foundation_model_path"])
    pretrained_foundation = os.path.exists(foundation_model_path)
    if not pretrained_foundation:
        raise FileNotFoundError(
            "❌ [CricaVPR] Missing foundation .pth file: "
            f"{foundation_model_path}. Please download dinov2_vitb14_pretrain.pth."
        )

    model = CricaVPRNet(
        pretrained_foundation=pretrained_foundation,
        foundation_model_path=foundation_model_path,
    )

    ckpt_path = _abspath(cfg["checkpoint"])
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            "❌ [CricaVPR] Missing checkpoint .pth file: "
            f"{ckpt_path}. Please download official CricaVPR.pth."
        )
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )
    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    vpr_logger.info(f"✅ [CricaVPR] Loaded checkpoint: {ckpt_path}")
    return model.to(device).eval()


@register("NetVLAD")
def _build_netvlad(cfg, device):
    sys.path.append(os.path.join(ROOT_DIR, "3rdparty", "Hierarchical-Localization"))
    from hloc import extractors
    from hloc.utils.base_model import dynamic_load

    Model = dynamic_load(extractors, "netvlad")
    inner = Model({"name": "netvlad"}).eval().to(device)

    class NetVLADWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m({"image": x})["global_descriptor"]

    return NetVLADWrapper(inner).eval()


@register("PairVPR")
def _build_pairvpr(cfg, device):
    pairvpr_path = _abspath(cfg.get("third_party", "3rdparty/Pair-VPR"))
    if pairvpr_path not in sys.path:
        sys.path.insert(0, pairvpr_path)

    import torch as _torch

    ckpt = _torch.load(
        _abspath(cfg["checkpoint"]), map_location="cpu", weights_only=False
    )
    from pairvpr.models.pairvpr import PairVPRNet
    from pairvpr.models.tools.pos_embed import interpolate_pos_embed
    from pairvpr.configs import pairvpr_speed

    model = PairVPRNet(pairvpr_speed).to(device)
    ckpt = {k: (v.to(device) if _torch.is_tensor(v) else v) for k, v in ckpt.items()}
    interpolate_pos_embed(pairvpr_speed, model, ckpt)
    model.load_state_dict(ckpt, strict=False)
    model.eval()

    class PairVPRWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, img):
            _, desc = self.m(img, None, mode="global")
            return desc

    vpr_logger.info(
        f"✅ [PairVPR] Loaded {os.path.basename(cfg['checkpoint'])}, "
        f"global desc = {cfg.get('feature_dim')}D"
    )
    return PairVPRWrapper(model)


__all__ = [
    "build_model",
    "get_transform",
    "register",
    "available_models",
    "ResizeToMultipleOf14",
]
