#!/usr/bin/env python


import argparse
import csv
import json
import os
import sys
from collections import OrderedDict, defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

POOL = "MegaLoc_8448"
TRAJ_GROUPS = [
    ("RobotCar", ["RobotCar"]),
    ("4Seasons", ["BusinessCampus", "Neighborhood", "OldTown"]),
    ("NCLT", ["NCLT"]),
    ("Boreas", ["Boreas"]),
    ("CustomCampus", ["CustomCampus"]),
]
SCENES = [s for _, g in TRAJ_GROUPS for s in g] + ["Pitts30k", "MSLS"]
BASELINES = {
    "pair_ft": "pairvpr_s2_target",
    "pair": "pairvpr_s2",
    "roma": "roma_gv_inliers",
}
LABELS = {
    "spare": "SPARE",
    "raw": "raw MegaLoc",
    "pair_ft": "Pair-VPR (fine-tuned)",
    "pair": "Pair-VPR (released)",
    "roma": "RoMa GV (inliers)",
}
CROPS = ["", "Ca4050", "Ca5060", "Ca7080", "Cfull"]


def _stem(scene):
    s = scene.replace("BusinessCampus", "business_campus").replace(
        "OldTown", "old_town"
    )
    return s.lower()


def _cache_path(scene):
    return os.path.join(ROOT, "assets", POOL, f"test_trajectories_{scene}_no_roi.json")


def _query_order(rows):
    order = OrderedDict()
    for r in rows:
        order.setdefault(str(r["query"]), None)
    return list(order)


def _spare_errors(scene):

    os.environ["DATASET"] = _stem(scene)
    for mod in [
        m for m in sys.modules if m == "core" or m.startswith(("core.", "sl_agent"))
    ]:
        del sys.modules[mod]
    import torch
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
        TOP_K,
        agent_input_dim,
        agent_tag,
        data_tag,
    )
    from sl_agent.vpr_classifier import (
        VPRReranking,
        build_extra_channels,
        build_sequence_features,
        center_frame_view,
        head_steps,
    )

    rows = json.load(open(_cache_path(scene), encoding="utf-8"))
    by_q = defaultdict(list)
    for item in rows:
        by_q[str(item["query"])].append(item)
    max_steps = sum(SPATIAL_STEPS) + 1
    window = head_steps(ABLATION_ARCH, max_steps)
    weight = DL_MODEL_SAVE_PATH.replace(
        ".pth", f"_{agent_tag()}_UnionMaps{data_tag()}.pth"
    )
    if not os.path.isabs(weight):
        weight = os.path.join(ROOT, weight)
    model = VPRReranking(
        input_dim=agent_input_dim(),
        hidden_dim=DL_HIDDEN_DIM,
        arch=ABLATION_ARCH,
        input_norm=ABLATION_INPUT_NORM,
    )
    model.load_state_dict(torch.load(weight, map_location="cpu", weights_only=True))
    model.eval()
    dummy = build_sequence_features(
        [],
        [],
        window,
        use_derivatives=ABLATION_USE_DERIVATIVES,
        use_progress=ABLATION_USE_PROGRESS,
        use_core=ABLATION_USE_CORE,
        use_scale=ABLATION_USE_SCALE,
    )
    out = []
    for q in _query_order(rows):
        cands = sorted(by_q[q], key=lambda x: x.get("rank", 999))[:TOP_K]
        head_cands = (
            center_frame_view(cands, max_steps) if window != max_steps else cands
        )
        extra = build_extra_channels(
            head_cands,
            window,
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
                window,
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
        out.append(float(cands[pick].get("error", -1)))
    raw = []
    for q in _query_order(rows):
        first = min(by_q[q], key=lambda x: x.get("rank", 999))
        raw.append(float(first.get("error", -1)))
    return np.array(out), np.array(raw)


def _baseline_errors(slug, scene, n):
    path = os.path.join(
        ROOT, "assets", "baseline_results", f"{slug}_{scene}__pool-{POOL}.json"
    )
    errs = json.load(open(path, encoding="utf-8"))["reranked_errors_m"]
    if len(errs) != n:
        raise SystemExit(f"{path}: {len(errs)} queries, cache has {n}")
    return np.array(errs, dtype=float)


def _hit(e, r):
    return ((e >= 0) & (e <= r)).astype(float)


def _fmt(lo, hi):
    return f"[{lo:+.1f}, {hi:+.1f}]"


def part_retrieval():
    print("\n## Paired bootstrap over test queries (seed-2026 universal head)")
    E = {}
    for s in SCENES:
        spare, raw = _spare_errors(s)
        E[s] = {"spare": spare, "raw": raw}
        for name, slug in BASELINES.items():
            E[s][name] = _baseline_errors(slug, s, len(raw))
    rng = np.random.default_rng(0)

    def boot(fn, B=4000):
        v = []
        for _ in range(B):
            idx = {
                s: rng.integers(0, len(E[s]["raw"]), len(E[s]["raw"])) for s in SCENES
            }
            v.append(fn(idx))
        return np.percentile(v, [2.5, 97.5])

    def traj(m, r, idx=None):
        out = []
        for _, grp in TRAJ_GROUPS:
            out.append(
                np.mean(
                    [
                        (
                            _hit(E[s][m], r)[idx[s]].mean()
                            if idx
                            else _hit(E[s][m], r).mean()
                        )
                        for s in grp
                    ]
                )
            )
        return 100 * np.mean(out)

    def one(s, m, r, idx=None):
        h = _hit(E[s][m], r)
        return 100 * (h[idx[s]].mean() if idx else h.mean())

    print("Trajectory mean (five datasets), SPARE minus baseline:")
    for b in ("raw", "pair_ft", "roma"):
        for r in (5, 10, 15):
            lo, hi = boot(lambda idx: traj("spare", r, idx) - traj(b, r, idx))
            d = traj("spare", r) - traj(b, r)
            print(f"  vs {LABELS[b]:<22} {r:>2} m: {d:+.1f}  95% CI {_fmt(lo, hi)}")
    print("Benchmarks, SPARE minus baseline:")
    for s, r, others in (
        ("Pitts30k", 25, ("raw", "pair", "pair_ft")),
        ("MSLS", 25, ("raw", "pair", "pair_ft")),
        ("MSLS", 15, ("pair_ft",)),
        ("Pitts30k", 10, ("roma",)),
        ("MSLS", 10, ("roma", "pair_ft")),
    ):
        for b in others:
            lo, hi = boot(lambda idx: one(s, "spare", r, idx) - one(s, b, r, idx))
            d = one(s, "spare", r) - one(s, b, r)
            print(
                f"  {s:<8} {r:>2} m vs {LABELS[b]:<22} {d:+.1f}  95% CI {_fmt(lo, hi)}"
            )
    print("Per dataset at 5 m, SPARE minus baseline:")
    for name, grp in TRAJ_GROUPS:

        def f(m, idx=None, grp=grp):
            return 100 * np.mean(
                [
                    _hit(E[s][m], 5)[idx[s]].mean() if idx else _hit(E[s][m], 5).mean()
                    for s in grp
                ]
            )

        for b in ("pair_ft", "roma"):
            lo, hi = boot(lambda idx: f("spare", idx) - f(b, idx))
            print(
                f"  {name:<12} vs {LABELS[b]:<22} {f('spare') - f(b):+.1f}  "
                f"95% CI {_fmt(lo, hi)}"
            )


def part_robot(B=20000):
    print("\n## Block bootstrap over the cold-start trials (20 contiguous time blocks)")
    path = os.path.join(ROOT, "results", "tables", "robot_trials_per_trial.csv")
    rows = sorted(
        csv.DictReader(open(path, encoding="utf-8")),
        key=lambda r: int(r["capture_time_ns"]),
    )
    arms = ("raw", "fol", "spare")
    ok = np.stack([np.array([float(r[f"{a}_success"]) for r in rows]) for a in arms])
    n = ok.shape[1]
    blocks = np.array_split(np.arange(n), 20)
    rng = np.random.default_rng(0)
    m = np.empty((B, len(arms)))
    for b in range(B):
        idx = np.concatenate(
            [blocks[i] for i in rng.integers(0, len(blocks), len(blocks))]
        )
        m[b] = 100 * ok[:, idx].mean(axis=1)
    names = {"raw": "raw MegaLoc", "fol": "FoL", "spare": "SPARE"}
    for k, a in enumerate(arms):
        lo, hi = np.percentile(m[:, k], [2.5, 97.5])
        print(
            f"  {names[a]:<12} {100 * ok[k].mean():.1f}% ({int(ok[k].sum())}/{n})  "
            f"95% CI [{lo:.1f}, {hi:.1f}]"
        )
    d = m[:, 2] - m[:, 1]
    lo, hi = np.percentile(d, [2.5, 97.5])
    print(
        f"  SPARE minus FoL {100 * (ok[2].mean() - ok[1].mean()):+.1f}  95% CI {_fmt(lo, hi)}"
    )


def part_fov(B=4000):
    print(
        "\n## Single-method intervals, field-of-view table (25 m), half-width in points"
    )
    rng = np.random.default_rng(0)
    for bench in ("Pitts30k", "MSLS"):
        widths = []
        for suf in CROPS:
            spare, raw = _spare_errors(bench + suf)
            for e in (raw, spare):
                h = _hit(e, 25)
                v = [h[rng.integers(0, len(h), len(h))].mean() * 100 for _ in range(B)]
                lo, hi = np.percentile(v, [2.5, 97.5])
                widths.append((hi - lo) / 2)
        print(f"  {bench:<8} half-width {min(widths):.1f}-{max(widths):.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="retrieval,robot,fov")
    a = ap.parse_args()
    parts = {"retrieval": part_retrieval, "robot": part_robot, "fov": part_fov}
    for p in [p.strip() for p in a.parts.split(",") if p.strip()]:
        parts[p]()


if __name__ == "__main__":
    main()
