#!/usr/bin/env python


import argparse
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
GEO_WARP = os.path.join(ROOT, "3rdparty", "geo_warp")
sys.path.insert(0, GEO_WARP)
os.environ.setdefault("DATASET", "robotcar")

from tools.rerank_eval_common import (
    add_eval_args,
    evaluate_scenes,
    parse_thresholds,
)


def build_scorer(arch, pooling, hr_weights, max_side):

    if not torch.cuda.is_available():
        raise SystemExit("GeoWarp's released code calls .cuda() unconditionally")

    import torchvision.transforms as transforms
    from PIL import Image
    import network
    import dataset_warp

    fe_ckpt = os.path.join(
        GEO_WARP, "data", "pretrained_baselines", f"{arch}_{pooling}.pth"
    )
    hr_ckpt = hr_weights or os.path.join(
        GEO_WARP, "data", "trained_homography_regressions", f"{arch}_{pooling}.pth"
    )
    for path in (fe_ckpt, hr_ckpt):
        if not os.path.isfile(path):
            raise SystemExit(
                f"missing GeoWarp weights: {path}\n"
                "run download_pretrained_baselines.py and download_trained_hom_reg.py "
                "inside 3rdparty/geo_warp"
            )

    features_extractor = network.FeaturesExtractor(arch, pooling)
    features_extractor.load_state_dict(torch.load(fe_ckpt, map_location="cpu"))
    homography_regression = network.HomographyRegression(
        kernel_sizes=[7, 5, 5, 5, 5, 5],
        channels=[225, 128, 128, 64, 64, 64, 64],
        padding=1,
    )
    homography_regression.load_state_dict(torch.load(hr_ckpt, map_location="cpu"))
    model = network.Network(features_extractor, homography_regression).cuda().eval()
    print(
        f"[geowarp] loaded {arch}+{pooling} feature extractor and "
        f"homography regression {os.path.basename(hr_ckpt)}"
    )

    normalize = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    def load_tensor(path):
        image = Image.open(path).convert("RGB")
        if max_side > 0:
            w, h = image.size
            scale = max_side / float(max(w, h))
            if scale < 1.0:
                image = image.resize(
                    (max(1, round(w * scale)), max(1, round(h * scale))), Image.BILINEAR
                )
        return normalize(image).unsqueeze(0)

    def extract(path):
        return path

    def score_candidates(query_path, candidate_paths):
        query = load_tensor(query_path).cuda()
        scores = []
        with torch.inference_mode():
            for cand_path in candidate_paths:
                cand = load_tensor(cand_path).cuda()
                warped_q, warped_c, _, _ = dataset_warp.compute_warping(
                    model, query, cand
                )
                q_feat = model("features_extractor", [warped_q, "local"])
                c_feat = model("features_extractor", [warped_c, "local"])
                scores.append(float((q_feat * c_feat).sum().item()))
        return scores

    return extract, score_candidates, [fe_ckpt, hr_ckpt]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--n-queries", type=int, default=0)
    ap.add_argument(
        "--arch", choices=("resnet50", "vgg16", "alexnet"), default="resnet50"
    )
    ap.add_argument("--pooling", choices=("netvlad", "gem"), default="netvlad")
    ap.add_argument(
        "--max-side",
        type=int,
        default=640,
        help="resize so the longer side is this many pixels before warping "
        "(GeoWarp was trained on 640x480 Pitts30k frames; 0 = native size)",
    )
    ap.add_argument(
        "--hr-weights",
        default=None,
        help="homography regression state_dict (tools/geowarp_target); "
        "default: the released weights",
    )
    ap.add_argument(
        "--method",
        default="GeoWarp-rerank",
        help="result name; use GeoWarp-target for the fine-tuned weights",
    )
    add_eval_args(ap)
    a = ap.parse_args()
    thresholds = parse_thresholds(a)

    from core.vpr_core import get_all_images_recursive

    extract, score_candidates, ckpts = build_scorer(
        a.arch, a.pooling, a.hr_weights, a.max_side
    )
    fe_ckpt, hr_ckpt = ckpts

    evaluate_scenes(
        ROOT,
        a,
        a.method,
        thresholds,
        [fe_ckpt, hr_ckpt],
        get_all_images_recursive,
        extract,
        score_candidates,
        {"arch": a.arch, "pooling": a.pooling, "max_side": a.max_side},
    )


if __name__ == "__main__":
    main()
