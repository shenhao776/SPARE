#!/usr/bin/env python


import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight-scene", default="UnionMaps")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch
    from collections import defaultdict
    from core.config import (
        ABLATION_ARCH,
        ABLATION_INPUT_NORM,
        ABLATION_USE_AREA,
        ABLATION_USE_CORE,
        ABLATION_USE_DERIVATIVES,
        ABLATION_USE_DINO,
        ABLATION_USE_HOMOG,
        ABLATION_USE_HULL,
        ABLATION_USE_POSE,
        ABLATION_USE_PROGRESS,
        ABLATION_USE_RANK,
        ABLATION_USE_SCALE,
        ABLATION_USE_SHAPE,
        ABLATION_USE_SIM,
        ABLATION_USE_WALK,
        ABLATION_USE_ZOOM,
        ABLATION_USE_ZOOMSHUF,
        DL_HIDDEN_DIM,
        DL_MODEL_SAVE_PATH,
        SPATIAL_STEPS,
        TARGET_SCENE,
        TOP_K,
        VPR_MODEL_ID,
        agent_input_dim,
        agent_tag,
        data_tag,
    )
    from core.paper_config import load_paper_config
    from sl_agent.vpr_classifier import (
        VPRReranking,
        build_extra_channels,
        build_sequence_features,
        center_frame_view,
        head_steps,
    )

    import json

    json_path = (
        f"assets/{VPR_MODEL_ID}/test_trajectories_{TARGET_SCENE}{data_tag()}.json"
    )
    raw = json.load(open(json_path, encoding="utf-8"))
    by_q = defaultdict(list)
    for item in raw:
        by_q[item["query"]].append(item)

    max_steps = sum(SPATIAL_STEPS) + 1
    head_window = head_steps(ABLATION_ARCH, max_steps)
    dim = agent_input_dim()
    seed0 = int(load_paper_config()["seeds"][0])
    seed = a.seed if a.seed is not None else seed0
    seed_suffix = f"_seed{seed}" if seed != seed0 else ""
    weight = DL_MODEL_SAVE_PATH.replace(
        ".pth", f"_{agent_tag()}_{a.weight_scene}{data_tag()}{seed_suffix}.pth"
    )
    model = VPRReranking(
        input_dim=dim,
        hidden_dim=DL_HIDDEN_DIM,
        arch=ABLATION_ARCH,
        input_norm=ABLATION_INPUT_NORM,
    )
    model.load_state_dict(torch.load(weight, map_location="cpu", weights_only=True))
    model.eval()
    dummy = build_sequence_features(
        [],
        [],
        head_window,
        use_derivatives=ABLATION_USE_DERIVATIVES,
        use_progress=ABLATION_USE_PROGRESS,
        use_core=ABLATION_USE_CORE,
        use_scale=ABLATION_USE_SCALE,
    )

    rows = []
    for q, items in by_q.items():
        cands = sorted(items, key=lambda x: x.get("rank", 999))[:TOP_K]
        if not cands:
            continue
        head_cands = (
            center_frame_view(cands, max_steps) if head_window != max_steps else cands
        )
        extra = build_extra_channels(
            head_cands,
            head_window,
            ABLATION_USE_SIM,
            ABLATION_USE_AREA,
            ABLATION_USE_RANK,
            ABLATION_USE_DINO,
            ABLATION_USE_POSE,
            ABLATION_USE_HULL,
            ABLATION_USE_HOMOG,
            ABLATION_USE_WALK,
            ABLATION_USE_SHAPE,
            ABLATION_USE_ZOOM,
            ABLATION_USE_ZOOMSHUF,
        )
        n_extra = len(extra[0]) if extra else 0
        feats = [
            build_sequence_features(
                head_cands[i].get("inliers", []),
                head_cands[i].get("scale_ratio", []),
                head_window,
                use_derivatives=ABLATION_USE_DERIVATIVES,
                use_progress=ABLATION_USE_PROGRESS,
                extra_channels=extra[i] if extra else None,
                use_core=ABLATION_USE_CORE,
                use_scale=ABLATION_USE_SCALE,
            )
            for i in range(len(cands))
        ]
        pad = [row + [0.0] * n_extra for row in dummy]
        while len(feats) < TOP_K:
            feats.append(pad)
        x = torch.tensor([feats], dtype=torch.float32)
        mask = torch.tensor([[1.0] * len(cands) + [0.0] * (TOP_K - len(cands))])
        with torch.no_grad():
            probs = model.predict_softmax(x, mask=mask)[0, :, -1].tolist()
        pick = max(range(len(cands)), key=lambda i: probs[i])
        errs = [float(c.get("error", -1)) for c in cands]
        best = min(range(len(cands)), key=lambda i: errs[i] if errs[i] >= 0 else 1e9)
        rows.append(
            {
                "query": os.path.basename(q),
                "raw_rank1_err_m": round(errs[0], 2),
                "pick_rank": pick + 1,
                "pick_err_m": round(errs[pick], 2),
                "pick_prob": round(probs[pick], 3),
                "best_rank": best + 1,
                "best_err_m": round(errs[best], 2),
                "pick_candidate": cands[pick].get("candidate_name"),
                "raw_candidate": cands[0].get("candidate_name"),
            }
        )
    rows.sort(key=lambda r: (-(r["raw_rank1_err_m"] - r["pick_err_m"])))
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    n_fix = sum(1 for r in rows if r["raw_rank1_err_m"] > 5 and r["pick_err_m"] <= 5)
    n_break = sum(1 for r in rows if r["raw_rank1_err_m"] <= 5 and r["pick_err_m"] > 5)
    print(
        f"{TARGET_SCENE}: {len(rows)} queries, fixed {n_fix}, broken {n_break} (5 m); weight {os.path.basename(weight)}"
    )
    print("top fixes (raw_err -> pick_err, pick rank):")
    for r in rows[:8]:
        print(
            f"  {r['query']}  {r['raw_rank1_err_m']:7.1f} -> {r['pick_err_m']:5.2f}  rank {r['pick_rank']}  p={r['pick_prob']}"
        )


if __name__ == "__main__":
    main()
