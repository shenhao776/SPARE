#!/usr/bin/env python


import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import cv2
from PIL import Image
import torchvision.transforms as transforms

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
TVPR = os.path.join(ROOT, "3rdparty", "TransVPR-model-implementation")
sys.path.insert(0, TVPR)
from tools.rerank_eval_common import (
    add_eval_args,
    matching_result,
    parse_thresholds,
    print_result,
    default_pool,
    result_metadata,
    result_path,
    save_result,
    scene_inputs,
)

from feature_extractor import Extractor_base
from blocks import POOL

IMG_SIZE = np.array([480, 640])
N_PATCH = IMG_SIZE // (2**4)


def _transform():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((int(IMG_SIZE[0]), int(IMG_SIZE[1]))),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_keypoints(img_size):
    H, W = img_size
    patch_size = 2**4
    N_h, N_w = H // patch_size, W // patch_size
    kps = np.zeros((2, N_h * N_w), dtype=int)
    kps[0] = np.tile(
        np.linspace(patch_size // 2, W - patch_size // 2, N_w, dtype=int), N_h
    )
    kps[1] = np.repeat(
        np.linspace(patch_size // 2, H - patch_size // 2, N_h, dtype=int), N_w
    )
    return np.transpose(kps)


def match_pair(fm1, fm2, mask1, mask2, s):

    M = torch.matmul(fm2, fm1.T)
    M[:, mask1 <= s] = 0
    M[mask2 <= s, :] = 0
    max1 = torch.argmax(M, dim=1)
    max2 = torch.argmax(M, dim=0)
    m = max2[max1]
    valid = torch.arange(M.shape[-1]).to(M.device) == m
    kps = get_keypoints(IMG_SIZE)
    idx1 = torch.nonzero(valid).squeeze()
    idx2 = max1[idx1]
    if idx1.ndim < 1 or idx1.numel() < 4:
        return 0.0
    _, mask = cv2.findHomography(
        kps[idx1.cpu().numpy()],
        kps[idx2.cpu().numpy()],
        cv2.FM_RANSAC,
        ransacReprojThreshold=(2**4) * 1.5,
    )
    return float(np.sum(mask)) if mask is not None else 0.0


def max_min_norm_tensor(mat):
    v_min, _ = torch.min(mat, -1, keepdims=True)
    v_max, _ = torch.max(mat, -1, keepdims=True)
    return (mat - v_min) / (v_max - v_min)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--n-queries", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.02)
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    add_eval_args(ap)
    a = ap.parse_args()
    thresholds = parse_thresholds(a)

    from core.vpr_core import get_all_images_recursive

    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    model = Extractor_base()
    pool = POOL(model.embedding_dim)
    model.add_module("pool", pool)
    ckpt_path = os.path.join(TVPR, "TransVPR_MSLS.pth")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt)
    model.to(device).eval()
    print(f"[transvpr] loaded MSLS checkpoint on {device}")

    tf = _transform()
    layer = a.layer
    shortlist = a.pool or default_pool()

    for scene in [s.strip() for s in a.scenes.split(",") if s.strip()]:
        path, work, db_paths = scene_inputs(
            ROOT, scene, a.n_queries, get_all_images_recursive, pool=shortlist
        )
        metadata = result_metadata(
            "TransVPR-rerank",
            scene,
            path,
            [ckpt_path],
            {
                "layer": a.layer,
                "attention_threshold": a.threshold,
                "n_queries_limit": int(a.n_queries),
            },
            pool=shortlist,
        )
        out_path = result_path(
            ROOT, a.output_dir, "TransVPR-rerank", scene, pool=shortlist
        )
        cached = None if a.force else matching_result(out_path, metadata, thresholds)
        if cached is not None:
            print(f"[resume] {scene}: matching result already exists")
            print_result(cached, out_path)
            continue

        def extract(path):
            img = tf(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
            with torch.inference_mode():
                feat = model(img)
                _, mask = model.pool(feat)
            patch = feat[:, layer, 1:, :][0]
            mask = torch.sum(max_min_norm_tensor(mask), 1)
            mask = max_min_norm_tensor(mask)[0]
            return patch.detach().cpu(), mask.detach().cpu()

        from collections import OrderedDict

        db_feat = OrderedDict()
        db_cache_cap = int(os.environ.get("RERANK_DB_CACHE_CAP", "2000"))

        def db_get(stem):
            if stem in db_feat:
                db_feat.move_to_end(stem)
                return db_feat[stem]
            value = extract(db_paths[stem])
            db_feat[stem] = value
            if len(db_feat) > db_cache_cap:
                db_feat.popitem(last=False)
            return value

        print(
            f"[{scene}] {len(db_paths)} candidate db frames (lazy LRU cache of {db_cache_cap})"
        )

        raw_errors, reranked_errors, selected_ranks = [], [], []
        t0 = time.time()
        for qi, (q, qp, resolved) in enumerate(work):
            cands = [candidate for candidate, _ in resolved]
            q_feat, q_mask = extract(qp)
            raw_errors.append(float(cands[0].get("error", -1)))
            best_score, best = -1, None
            for c, stem in resolved:
                d_feat, d_mask = db_get(stem)
                score = match_pair(
                    q_feat.to(device),
                    d_feat.to(device),
                    q_mask.to(device),
                    d_mask.to(device),
                    a.threshold,
                )
                if score > best_score:
                    best_score, best = score, c
            if best is None:
                raise RuntimeError(f"{scene}/{q}: no candidate received a score")
            reranked_errors.append(float(best.get("error", -1)))
            selected_ranks.append(int(best.get("rank", -1)))
            if qi % 50 == 0:
                print(f"  [{scene}] {qi}/{len(work)} {time.time()-t0:.0f}s", flush=True)
        result = save_result(
            out_path,
            metadata,
            thresholds,
            raw_errors,
            reranked_errors,
            selected_ranks,
            time.time() - t0,
        )
        print_result(result, out_path)


if __name__ == "__main__":
    main()
