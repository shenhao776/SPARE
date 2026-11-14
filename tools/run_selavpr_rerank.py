#!/usr/bin/env python


import argparse
import os
import sys
import time
from collections import OrderedDict
from types import SimpleNamespace

import torch
import torchvision.transforms as transforms
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
SELA = os.path.join(ROOT, "3rdparty", "SelaVPR")
from tools.rerank_eval_common import (
    add_eval_args,
    default_pool,
    matching_result,
    parse_thresholds,
    print_result,
    result_metadata,
    result_path,
    save_result,
    scene_inputs,
)

RESIZE = (224, 224)


def load_model(ckpt_path, device):
    sys.path.insert(0, SELA)
    import network
    from local_matching import local_sim

    args = SimpleNamespace(
        registers=False, foundation_model_path=None, resume=ckpt_path
    )
    model = network.GeoLocalizationNet(args)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)[
        "model_state_dict"
    ]
    state = {
        k[len("module.") :] if k.startswith("module.") else k: v
        for k, v in state.items()
    }
    model.load_state_dict(state)
    return model.to(device).eval(), local_sim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--n-queries", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--weights", default=os.path.join(SELA, "weights", "SelaVPR_msls.pth")
    )
    ap.add_argument(
        "--method",
        default="SelaVPR-rerank",
        help="result name; use SelaVPR-target for the fine-tuned weights",
    )
    add_eval_args(ap)
    a = ap.parse_args()
    thresholds = parse_thresholds(a)

    from core.vpr_core import get_all_images_recursive

    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    model, local_sim = load_model(a.weights, device)
    print(f"[selavpr] loaded {os.path.basename(a.weights)} on {device}")
    tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    pool = a.pool or default_pool()

    def extract(path):
        img = Image.open(path).convert("RGB")
        img = transforms.functional.resize(img, list(RESIZE))
        with torch.inference_mode():
            local, _global = model(tf(img).unsqueeze(0).to(device))
        return local[0].detach()

    for scene in [s.strip() for s in a.scenes.split(",") if s.strip()]:
        path, work, db_paths = scene_inputs(
            ROOT, scene, a.n_queries, get_all_images_recursive, pool=pool
        )
        metadata = result_metadata(
            a.method,
            scene,
            path,
            [a.weights],
            {
                "resize": list(RESIZE),
                "score": "mutual-NN count",
                "n_queries_limit": int(a.n_queries),
            },
            pool=pool,
        )
        out_path = result_path(ROOT, a.output_dir, a.method, scene, pool=pool)
        cached = None if a.force else matching_result(out_path, metadata, thresholds)
        if cached is not None:
            print(f"[resume] {scene}: matching result already exists")
            print_result(cached, out_path)
            continue

        db_feat = OrderedDict()
        cap = int(os.environ.get("RERANK_DB_CACHE_CAP", "1000"))

        def db_get(stem):
            if stem in db_feat:
                db_feat.move_to_end(stem)
                return db_feat[stem]
            value = extract(db_paths[stem])
            db_feat[stem] = value
            if len(db_feat) > cap:
                db_feat.popitem(last=False)
            return value

        raw_errors, reranked_errors, selected_ranks = [], [], []
        t0 = time.time()
        for qi, (q, qp, resolved) in enumerate(work):
            cands = [candidate for candidate, _ in resolved]
            q_feat = extract(qp)
            c_feat = torch.stack([db_get(stem) for _, stem in resolved])
            with torch.inference_mode():
                scores = local_sim(q_feat.unsqueeze(0), c_feat).cpu().tolist()
            best_i = max(range(len(scores)), key=lambda i: (scores[i], -i))
            best = cands[best_i]
            raw_errors.append(float(cands[0].get("error", -1)))
            reranked_errors.append(float(best.get("error", -1)))
            selected_ranks.append(int(best.get("rank", -1)))
            if qi % 100 == 0:
                print(
                    f"  [{scene}] {qi}/{len(work)} {time.time() - t0:.0f}s", flush=True
                )
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
