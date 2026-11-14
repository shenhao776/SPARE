#!/usr/bin/env python


import argparse
import os
import sys

import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
PVPR = os.path.join(ROOT, "3rdparty", "Pair-VPR")
sys.path.insert(0, PVPR)

from tools.rerank_eval_common import (
    add_eval_args,
    evaluate_scenes,
    parse_thresholds,
)


def build_model(device, checkpoint=None):
    import torch as _torch
    from pairvpr.models.pairvpr import PairVPRNet
    from pairvpr.models.tools import blocks as _pairvpr_blocks
    from pairvpr.models.tools.pos_embed import interpolate_pos_embed
    from pairvpr.configs import pairvpr_speed

    if not _pairvpr_blocks.XFORMERS_AVAILABLE:

        def _compat_mem_eff_cross_attention(
            self, query, key, value, qpos=None, kpos=None, attn_bias=None
        ):
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return _pairvpr_blocks.CrossAttention.forward(self, query, key, value)

        _pairvpr_blocks.MemEffCrossAttention.forward = _compat_mem_eff_cross_attention

    released = os.path.join(ROOT, "assets", "PairVPR_512", "pairvpr-vitB.pth")
    ckpt = _torch.load(checkpoint or released, map_location="cpu", weights_only=False)
    model = PairVPRNet(pairvpr_speed).to(device)
    ckpt = {k: (v.to(device) if _torch.is_tensor(v) else v) for k, v in ckpt.items()}
    interpolate_pos_embed(pairvpr_speed, model, ckpt)
    result = model.load_state_dict(ckpt, strict=False)
    if checkpoint and (result.missing_keys or result.unexpected_keys):
        raise SystemExit(
            f"[pairvpr-s2] {checkpoint}: {len(result.missing_keys)} missing / "
            f"{len(result.unexpected_keys)} unexpected keys"
        )
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--n-queries", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--checkpoint",
        default=None,
        help="fine-tuned state_dict (tools/pairvpr_target); default: released vitB",
    )
    ap.add_argument(
        "--method",
        default="PairVPR-s2",
        help="result name; use PairVPR-s2-target for the fine-tuned weights",
    )
    add_eval_args(ap)
    a = ap.parse_args()
    thresholds = parse_thresholds(a)

    from core.vpr_core import get_all_images_recursive
    import torchvision.transforms as transforms

    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    model = build_model(device, a.checkpoint)
    checkpoint = a.checkpoint or os.path.join(
        ROOT, "assets", "PairVPR_512", "pairvpr-vitB.pth"
    )
    print(f"[pairvpr-s2] loaded pairvpr-vitB stage-2 on {device}")
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(
                (322, 322),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    def dense(path):
        image = transform(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        with torch.inference_mode():
            features, _ = model(image, None, mode="global")
        return features[0].cpu()

    def score_candidates(query_feature, candidate_features):
        scores = []
        query = query_feature.unsqueeze(0).to(device)
        with torch.inference_mode():
            for candidate_feature in candidate_features:
                database = candidate_feature.unsqueeze(0).to(device)
                score = model(query, database, "pairvpr") + model(
                    database, query, "pairvpr"
                )
                scores.append(score.item())
        return scores

    evaluate_scenes(
        ROOT,
        a,
        a.method,
        thresholds,
        [checkpoint],
        get_all_images_recursive,
        dense,
        score_candidates,
        {
            "resize": [322, 322],
            "symmetric_scoring": True,
        },
    )


if __name__ == "__main__":
    main()
