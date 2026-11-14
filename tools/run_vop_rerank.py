#!/usr/bin/env python


import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
VOP = os.path.join(ROOT, "3rdparty", "vop")
sys.path.insert(0, VOP)
os.chdir(VOP)

from tools.rerank_eval_common import (
    add_eval_args,
    evaluate_scenes,
    parse_thresholds,
)

IMSIZE = 224
RADIUS = -1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--n-queries", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--radius", type=float, default=RADIUS)
    add_eval_args(ap)
    a = ap.parse_args()
    thresholds = parse_thresholds(a)

    from core.vpr_core import get_all_images_recursive
    from evaluate_utils import Voting
    from lightglue.utils import load_image

    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    import gluefactory

    model = gluefactory.load_experiment("demo/checkpoint_best.tar").to(device).eval()
    print(f"[vop] loaded Breaking-the-Frame checkpoint on {device}")

    def patch_des(path):
        img = load_image(path, resize=IMSIZE).to(device)
        _, height, width = img.shape
        pad = torch.zeros((3, IMSIZE, IMSIZE), device=device, dtype=img.dtype)
        pad[:, :height, :width] = img
        with torch.inference_mode():
            feats = model.extractor({"image": pad[None]})
            feats["image_size"] = [[float(width), float(height)]]
            data = {}
            for key, value in feats.items():
                tensor = (
                    value.to(device)
                    if torch.is_tensor(value)
                    else torch.as_tensor(value, device=device)
                )
                data[key + "0"] = tensor
                data[key + "1"] = tensor
            pred = model.matcher(data)
        return pred["desc0"][0].detach().cpu()

    state = {"radius": a.radius}

    def per_scene_setup(scene, work, db_features, extract_fn):

        if a.radius != -1.0:
            state["radius"] = a.radius
            return
        q_sample = [extract_fn(qp) for _, qp, _ in work[:50]]
        db_sample = list(db_features.values())[:50]
        sample = torch.stack(q_sample + db_sample)
        sims = torch.nn.functional.cosine_similarity(
            sample.unsqueeze(2), sample.unsqueeze(1), dim=-1
        )
        state["radius"] = round((torch.round(sims.median() * 10) / 10).item(), 2)
        print(f"[vop] {scene}: auto-median radius = {state['radius']}")

    def score_candidates(query_feature, candidate_features):
        database = torch.stack(candidate_features).to(device)
        voting = Voting(state["radius"], num_patches=database.shape[1])
        candidate_ids = torch.arange(len(candidate_features), device=device).unsqueeze(
            0
        )
        with torch.inference_mode():
            output = voting.rerank(
                query_feature.unsqueeze(0).to(device),
                database,
                candidate_ids,
            )["scores"]
        scores = output[1, 0] if np.any(output[1, 0] != 0) else output[0, 0]
        return np.asarray(scores).reshape(-1).tolist()

    evaluate_scenes(
        ROOT,
        a,
        "VOP-rerank",
        thresholds,
        [os.path.join(VOP, "demo", "checkpoint_best.tar")],
        get_all_images_recursive,
        patch_des,
        score_candidates,
        {
            "resize": IMSIZE,
            "radius": a.radius,
            "image_size_convention": "pre-pad [w,h]",
        },
        per_scene_setup=per_scene_setup,
    )


if __name__ == "__main__":
    main()
