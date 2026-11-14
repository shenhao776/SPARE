#!/usr/bin/env python


import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True, help="query stem (nanosecond timestamp)")
    ap.add_argument("--weight-scene", default="UnionMaps")
    ap.add_argument("--seed", type=int, default=None, help="default: paper seed 0")
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
        ABLATION_USE_VPRPATCH,
        ABLATION_USE_VPRPATCHSHUF,
        DL_MODEL_SAVE_PATH as BASE_MODEL_SAVE_PATH,
        DL_HIDDEN_DIM,
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

    json_path = (
        f"assets/{VPR_MODEL_ID}/test_trajectories_{TARGET_SCENE}{data_tag()}.json"
    )
    raw = json.load(open(json_path, encoding="utf-8"))
    by_q = defaultdict(list)
    for item in raw:
        by_q[item["query"]].append(item)
    keys = [q for q in by_q if a.query in os.path.basename(q)]
    if len(keys) != 1:
        raise SystemExit(f"query {a.query}: {len(keys)} matches in {json_path}")
    cands = sorted(by_q[keys[0]], key=lambda x: x.get("rank", 999))[:TOP_K]

    max_steps = sum(SPATIAL_STEPS) + 1
    head_window = head_steps(ABLATION_ARCH, max_steps)
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
    feats = []
    for ci in range(len(cands)):
        feats.append(
            build_sequence_features(
                head_cands[ci].get("inliers", []),
                head_cands[ci].get("scale_ratio", []),
                head_window,
                use_derivatives=ABLATION_USE_DERIVATIVES,
                use_progress=ABLATION_USE_PROGRESS,
                extra_channels=extra[ci] if extra else None,
                use_core=ABLATION_USE_CORE,
                use_scale=ABLATION_USE_SCALE,
            )
        )
    dim = agent_input_dim()
    assert all(len(f) == head_window and len(f[0]) == dim for f in feats), (
        dim,
        head_window,
    )

    seed0 = int(load_paper_config()["seeds"][0])
    seed = a.seed if a.seed is not None else seed0
    seed_suffix = f"_seed{seed}" if seed != seed0 else ""
    weight = BASE_MODEL_SAVE_PATH.replace(
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
    x = torch.tensor([feats], dtype=torch.float32)
    mask = torch.ones(1, len(cands))
    with torch.no_grad():
        probs = model.predict_softmax(x, mask=mask)[0, :, -1].tolist()
    pick = max(range(len(probs)), key=lambda i: probs[i])

    names = []
    if ABLATION_USE_CORE:
        names.append("log inliers")
        if ABLATION_USE_SCALE:
            names.append("hull scale ratio")
    if ABLATION_USE_DERIVATIVES:
        names += ["d scale", "d2 scale"]
    if ABLATION_USE_PROGRESS:
        names.append("progress")
    if ABLATION_USE_SIM:
        names.append("retrieval sim")
    if ABLATION_USE_AREA:
        names += ["hull area q", "hull area db"]
    if ABLATION_USE_RANK:
        names.append("retrieval rank")
    if ABLATION_USE_DINO:
        names.append("DINOv2 agreement")
    if ABLATION_USE_POSE:
        names += ["signed dist", "heading diff"]
    if ABLATION_USE_HOMOG:
        names += ["H tx", "H ty", "H log mag", "H perspective"]
    if ABLATION_USE_ZOOM:
        names.append("magnification")
    if ABLATION_USE_VPRPATCH:
        names.append("VPR patch agreement")
    if ABLATION_USE_VPRPATCHSHUF:
        names.append("VPR patch agreement (shuffled)")
    assert len(names) == dim, (names, dim)

    out = {
        "query": os.path.basename(keys[0]),
        "query_path": keys[0],
        "scene": TARGET_SCENE,
        "candidates": [
            c.get("rel_path") or c.get("path") or c.get("name") for c in cands
        ],
        "errors_m": [c.get("error") for c in cands],
        "ranks": [c.get("rank") for c in cands],
        "state_matrix_rows_candidates_cols_channels": [f[0] for f in feats],
        "channels": names,
        "softmax": probs,
        "pick_index": pick,
        "weight": os.path.basename(weight),
        "note": (
            f"{TARGET_SCENE} test query {os.path.basename(keys[0])}: single-frame state "
            f"(centre frame of the cached window, K x 1 min-max blocks) built by "
            f"sl_agent.vpr_classifier exactly as in training/eval; softmax from the "
            f"{a.weight_scene} head (seed {seed}). Raw rank 1 error "
            f"{cands[0].get('error'):.2f} m, picked rank {pick + 1} error {cands[pick].get('error'):.2f} m."
        ),
    }
    os.makedirs(os.path.dirname(os.path.expanduser(a.out)) or ".", exist_ok=True)
    json.dump(out, open(os.path.expanduser(a.out), "w", encoding="utf-8"), indent=1)
    print(out["note"])
    print("channels:", names)
    print("softmax:", [round(p, 3) for p in probs])


if __name__ == "__main__":
    main()
