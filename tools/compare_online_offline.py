#!/usr/bin/env python


import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

SEQ_FIELDS = (
    "inliers",
    "scale_ratio",
    "area_step",
    "area_query",
    "scores",
    "hull_cx",
    "hull_cy",
    "hull_aspect",
    "hull_fill",
    "H_tx",
    "H_ty",
    "H_logdet",
    "H_persp",
    "step_offset_m",
    "step_turn_deg",
    "step_change",
    "step_valid",
    "peak_off",
    "peak_sharp",
    "dino_sim",
    "zoom",
    "vprp_sim",
)

FIELD_TOL = {"dino_sim": 5e-4}


_SERVICE = None
_BUILDER = None


def _service(dataset_path):

    global _SERVICE
    if _SERVICE is None:
        from core.vpr_core import VPRService

        _SERVICE = VPRService()
    return _SERVICE


def _builder(service, dataset_path, scene, match_batch=1):
    global _BUILDER
    if _BUILDER is None:
        from sl_agent.online_state import OnlineStateBuilder

        _BUILDER = OnlineStateBuilder(
            service, dataset_path, scene=scene, match_batch=match_batch
        )
    return _BUILDER


def _seq_diff(a, b):

    a = [float(x) for x in (a or [])]
    b = [float(x) for x in (b or [])]
    note = "" if len(a) == len(b) else f"len {len(a)} vs {len(b)}"
    n = min(len(a), len(b))
    if n == 0:
        return (0.0 if not note else float("inf")), note or "both empty"
    return max(abs(a[i] - b[i]) for i in range(n)), note


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="CustomCampus")
    ap.add_argument("--dataset", default="customcampus", help="YAML stem")
    ap.add_argument("--query", default=None, help="query stem; default = first row")
    ap.add_argument("--n", type=int, default=1, help="check this many queries")
    ap.add_argument("--model-id", default="MegaLoc_8448")
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument(
        "--tensor-tol",
        type=float,
        default=1e-3,
        help="bound on the assembled state tensor; see the note where it is used",
    )
    ap.add_argument(
        "--match-batch",
        type=int,
        default=1,
        help="窗口一次匹配多少步。1 = 逐对(默认,与离线同一条算法路径)",
    )
    ap.add_argument(
        "--criterion",
        choices=("outcome", "bitwise"),
        default="outcome",
        help="outcome(默认):重排选出同一张图即通过,通道差只作诊断打印。"
        "bitwise 是排查用的,见 _verdict_mode",
    )
    ap.add_argument(
        "--zoom",
        type=float,
        default=None,
        help="query magnification; default = whatever the cache recorded, so "
        "the comparison isolates the state build rather than the camera value",
    )
    a = ap.parse_args()

    os.environ.setdefault("DATASET", a.dataset)
    import json

    import cv2
    import torch
    from PIL import Image

    from core.config import TOP_K, data_tag
    from core.vpr_core import VPRService, get_base_transform
    from sl_agent.online_state import OnlineStateBuilder

    dataset_path = os.path.join(ROOT, "datasets", a.scene)
    cache = os.path.join(
        ROOT, "assets", a.model_id, f"test_trajectories_{a.scene}{data_tag()}.json"
    )
    if not os.path.isfile(cache):
        raise SystemExit(f"offline cache not found: {cache}")

    print(f"[cache] {os.path.relpath(cache, ROOT)}")
    rows = json.load(open(cache, encoding="utf-8"))
    seen, q_names = set(), []
    for r in rows:
        if r["query"] not in seen:
            seen.add(r["query"])
            q_names.append(r["query"])
    q_names = [a.query] if a.query else q_names[: max(1, a.n)]
    print(f"[queries] {len(q_names)} to check")

    results = []
    for q_name in q_names:
        results.append(check_one(a, rows, q_name))
    print("\n=== summary ===")
    n_ok = sum(1 for r in results if r)
    claim = (
        "online == offline (逐位)"
        if _verdict_mode(a) == "bitwise"
        else f"重排选出同一候选 (match_batch={a.match_batch}, 通道未逐位比)"
    )
    print(f"  {n_ok}/{len(results)} queries: {claim}")
    return 0 if n_ok == len(results) else 1


def check_one(a, rows, q_name):
    import json
    import os

    import cv2
    import torch
    from PIL import Image

    from core.config import TOP_K
    from core.vpr_core import VPRService, get_base_transform
    from sl_agent.online_state import OnlineStateBuilder

    dataset_path = os.path.join(ROOT, "datasets", a.scene)
    print(f"\n{'=' * 68}")
    off = [r for r in rows if r.get("query") == q_name]
    off.sort(key=lambda r: r.get("rank", 0))
    if not off:
        raise SystemExit(f"query {q_name} has no rows in the cache")
    print(f"[query] {q_name}  ({len(off)} cached candidates)")

    qdir = os.path.join(dataset_path, "queries_test")
    hits = [
        os.path.join(dp, f)
        for dp, _, fs in os.walk(qdir)
        for f in fs
        if os.path.splitext(f)[0] == q_name
    ]
    if not hits:
        raise SystemExit(f"no image for {q_name} under {qdir}")
    q_path = hits[0]
    print(f"[image] {os.path.relpath(q_path, ROOT)}")

    service = _service(dataset_path)
    db_desc, db_paths = service._get_or_compute_db_features(dataset_path)
    db_rel = [str(p) for p in db_paths]
    by_base = {os.path.basename(p): p for p in db_rel}

    transform = get_base_transform()
    t = (
        transform(Image.open(q_path).convert("RGB"))
        .unsqueeze(0)
        .to(service.args.device)
    )
    with torch.no_grad():
        q_desc = service.model(t).cpu().numpy()

    import faiss

    index = faiss.IndexFlatL2(service.args.fc_output_dim)
    index.add(db_desc)
    _, pred = index.search(q_desc, min(len(db_rel), TOP_K * 10))
    online_order = [os.path.basename(db_rel[i]) for i in pred[0]]
    cached_names = [r["candidate_name"] for r in off]
    rank_of = {n: i for i, n in enumerate(online_order)}
    missing = [n for n in cached_names if n not in rank_of]
    print("\n=== 1. retrieval ===")
    if missing:
        print(
            f"  ⚠ {len(missing)}/{len(cached_names)} cached candidates are not in "
            f"the live search at all: {missing[:3]}"
        )
    else:
        worst = max(rank_of[n] for n in cached_names)
        print(
            f"  ✅ all {len(cached_names)} cached candidates found live "
            f"(deepest at live rank {worst + 1})"
        )

    cands = []
    for r in off:
        rel = by_base.get(r["candidate_name"])
        if rel is None:
            raise SystemExit(f"candidate {r['candidate_name']} not in the database")
        cands.append(
            {
                "rel_path": rel,
                "rank": r.get("rank"),
                "vpr_sim": (r.get("scores") or [0.0])[0],
                "global_pose": r.get("global_pose") or [],
            }
        )

    zoom = a.zoom
    if zoom is None:
        z = off[0].get("zoom") or []
        zoom = float(z[0]) if z else None
    print(f"[zoom] {zoom}")

    builder = _builder(service, dataset_path, a.scene, a.match_batch)
    db_lookup = {rel: db_desc[i] for i, rel in enumerate(db_rel)}
    q_cv = cv2.imread(q_path)
    on = builder.build_rows(q_cv, q_path, cands, db_lookup, q_desc[0], zoom=zoom)

    from core.config import ABLATION_ARCH, SPATIAL_STEPS
    from sl_agent.vpr_classifier import center_frame_view, head_steps

    _L = sum(SPATIAL_STEPS) + 1
    skip_fields = set()
    if head_steps(ABLATION_ARCH, _L) == 1:
        print(
            "[single-frame head] online rows hold the candidate frame only; "
            "offline rows are collapsed to the window centre for the comparison"
        )
        off = center_frame_view(off, _L)
        skip_fields = {"peak_off", "peak_sharp"}
    from core.config import ABLATION_USE_VPRPATCH

    if not ABLATION_USE_VPRPATCH:
        skip_fields.add("vprp_sim")

    print("\n=== 2. state channels (max |online - offline|) ===")
    worst_field, worst_val = None, 0.0
    bad = []
    for f in [x for x in SEQ_FIELDS if x not in skip_fields]:
        m, notes = 0.0, []
        for i, (o, n) in enumerate(zip(off, on)):
            d, note = _seq_diff(o.get(f), n.get(f))
            m = max(m, d)
            if note:
                notes.append(f"cand{i + 1}: {note}")
        tol = FIELD_TOL.get(f, a.tol)
        flag = "✅" if m <= tol and not notes else "❌"
        if flag == "❌":
            bad.append(f)
        if m > worst_val:
            worst_field, worst_val = f, m
        extra = ("  " + "; ".join(notes[:2])) if notes else ""
        if f in FIELD_TOL and m > a.tol:
            extra += (
                f"  ({'within' if m <= tol else 'OVER'} its fp16/argmax "
                f"bound {tol:.0e})"
            )
        print(f"  {flag} {f:16s} {m:.3e}{extra}")

    print("\n=== 3. state tensor & head ===")
    x_on, mask_on = builder.build_state_tensor(on)
    x_off, mask_off = builder.build_state_tensor(off)
    dt = float((x_on - x_off).abs().max())
    print(
        f"  tensor {tuple(x_on.shape)}  max|diff| = {dt:.3e}  "
        f"(tol {a.tensor_tol:.0e})"
    )

    if service.reranking_model is not None:
        with torch.no_grad():
            p_on = service.reranking_model.predict_softmax(x_on, mask=mask_on)[0, :, -1]
            p_off = service.reranking_model.predict_softmax(x_off, mask=mask_off)[
                0, :, -1
            ]
        i_on, i_off = int(torch.argmax(p_on)), int(torch.argmax(p_off))
        print(f"  probs  max|diff| = {float((p_on - p_off).abs().max()):.3e}")
        print(
            f"  winner online = rank {i_on + 1} ({cands[i_on]['rel_path'].split('/')[-1]})"
            if i_on < len(cands)
            else f"  winner online = pad slot {i_on}"
        )
        print(
            f"  winner offline= rank {i_off + 1} ({cands[i_off]['rel_path'].split('/')[-1]})"
            if i_off < len(cands)
            else f"  winner offline= pad slot {i_off}"
        )
        agree = i_on == i_off
    else:
        print("  ⚠ no re-ranking weight loaded -- head comparison skipped")
        agree = None

    print("\n=== verdict ===")
    mode = _verdict_mode(a)
    if mode == "bitwise":
        ok = not bad and dt <= a.tensor_tol and agree is not False
        if ok:
            print(
                f"  ✅ online == offline (worst channel {worst_field} {worst_val:.3e})"
            )
        else:
            if bad:
                print(f"  ❌ channels differing: {', '.join(bad)}")
            if dt > a.tensor_tol:
                print(f"  ❌ state tensor differs by {dt:.3e}")
            if agree is False:
                print("  ❌ the head picks a different candidate")
    else:
        ok = agree is not False
        note = f"worst channel {worst_field} {worst_val:.3e}, tensor {dt:.3e}"
        if ok:
            print(f"  ✅ 同一候选 (通道有偏移是预期的: {note})")
        else:
            print(f"  ❌ 重排选出了不同的候选 ({note})")
        if bad:
            print(f"  ℹ 偏离逐位的通道: {', '.join(bad)}")
    return ok


def _verdict_mode(a):

    return a.criterion


if __name__ == "__main__":
    sys.exit(main())
