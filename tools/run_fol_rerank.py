#!/usr/bin/env python


import argparse
import os
import sys

import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
FOL = os.path.join(ROOT, "3rdparty", "FoL")
sys.path.insert(0, FOL)

from tools.rerank_eval_common import (
    add_eval_args,
    evaluate_scenes,
    parse_thresholds,
)

T2 = 0.7


def match_batch(fm1, fm2):

    M = torch.matmul(fm2, fm1.T)
    max1 = torch.argmax(M, dim=1)
    max2 = torch.argmax(M, dim=2)
    batch_indexes = torch.arange(M.shape[0], device=M.device).reshape((-1, 1))
    m = max2[batch_indexes, max1]
    valid = torch.arange(M.shape[-1], device=M.device).repeat((M.shape[0], 1)) == m
    scores = torch.zeros(fm2.shape[0], device=M.device)
    for i in range(fm2.shape[0]):
        idx1 = torch.nonzero(valid[i, :]).squeeze()
        idx2 = max1[i, :][idx1]
        if len(idx1.shape) > 0:
            cos = torch.sum(fm1[idx1] * fm2[i][idx2], dim=1)
            idx1 = idx1[cos > T2]
            idx2 = idx2[cos > T2]
        scores[i] = len(idx1) if len(idx1.shape) >= 1 else 0
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--n-queries", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--backbone", choices=("vitl14", "vitb14"), default="vitl14")
    add_eval_args(ap)
    a = ap.parse_args()
    thresholds = parse_thresholds(a)

    from core.vpr_core import get_all_images_recursive
    import torchvision.transforms as transforms
    from hubconf import fol_vitb14, fol_vitl14

    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    if a.backbone == "vitl14":
        model = fol_vitl14(pretrained=True).to(device).eval()
        ckpt_name = "FoL_large.pth"
    else:
        model = fol_vitb14(pretrained=True).to(device).eval()
        ckpt_name = "FoL_base.pth"
    checkpoint = os.path.join(torch.hub.get_dir(), "checkpoints", ckpt_name)
    print(f"[fol] loaded FoL {a.backbone} (AAAI 2025) on {device}")
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((504, 504)),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    def local_feat(path):
        image = transform(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        with torch.inference_mode():
            output = model(image, test=True)
        return output[1][0].cpu()

    def score_candidates(query_feature, candidate_features):
        database = torch.stack(candidate_features).to(device)
        with torch.inference_mode():
            scores = match_batch(query_feature.to(device), database)
        return scores.detach().cpu().tolist()

    evaluate_scenes(
        ROOT,
        a,
        "FoL-rerank",
        thresholds,
        [checkpoint],
        get_all_images_recursive,
        local_feat,
        score_candidates,
        {
            "resize": [504, 504],
            "cosine_threshold": T2,
            "backbone": a.backbone,
        },
    )


if __name__ == "__main__":
    main()
