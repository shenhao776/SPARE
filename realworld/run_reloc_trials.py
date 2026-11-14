#!/usr/bin/env python


import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)


def _link(dst, src):
    if os.path.islink(dst):
        if os.path.realpath(dst) == os.path.realpath(src):
            return
        os.unlink(dst)
    elif os.path.exists(dst):
        raise SystemExit(f"{dst} 存在但不是软链,请人工确认后删除")
    os.symlink(src, dst)


def _replay_dataset(scene):

    live = os.path.join(ROOT, "realworld", "live", f"{scene}_reloc")
    os.makedirs(live, exist_ok=True)
    _link(
        os.path.join(live, "database"),
        os.path.join(ROOT, "datasets", scene, "database"),
    )
    _link(
        os.path.join(live, "queries_test"),
        os.path.join(ROOT, "datasets", scene, "queries_test"),
    )
    return live


def load_pose7(pose_txt):

    out = {}
    with open(pose_txt, encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) < 8:
                continue
            from decimal import Decimal

            out[int(Decimal(p[0]) * Decimal("1e9"))] = [float(x) for x in p[1:8]]
    return out


def pose7_of_image(img_path, cache):

    seq = os.path.dirname(os.path.dirname(os.path.abspath(img_path)))
    if seq not in cache:
        cache[seq] = load_pose7(os.path.join(seq, "pose.txt"))
        cache[seq + "@keys"] = np.array(sorted(cache[seq]), dtype=np.int64)
    ns = int(os.path.splitext(os.path.basename(img_path))[0].split("@")[0])
    if ns in cache[seq]:
        return cache[seq][ns]
    keys = cache[seq + "@keys"]
    j = int(np.argmin(np.abs(keys - ns)))
    if abs(int(keys[j]) - ns) > 250_000_000:
        return None
    return cache[seq][int(keys[j])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="CustomCampus")
    ap.add_argument(
        "--trials",
        default=os.path.join(ROOT, "realworld", "reloc_frames", "trials.json"),
    )
    ap.add_argument("--out", default=os.path.join(ROOT, "realworld", "reloc_results"))
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 个,调试用")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    trials = json.load(open(a.trials, encoding="utf-8"))
    if a.limit:
        trials = trials[: a.limit]
    print(f"[trials] {len(trials)} 次试验,来自 {a.trials}")

    os.environ.setdefault("DATASET", "customcampus")
    from core.vpr_core import VPRService

    service = VPRService()
    active = getattr(service, "reranking_model", None) is not None
    print(f"[rerank_active] {active}")
    if not active:
        raise SystemExit("重排器没加载 —— star 臂会静默等价于 raw,拒绝继续。")

    ds = _replay_dataset(a.scene)
    print(f"[dataset] {os.path.relpath(ds, ROOT)}")

    pose_cache = {}
    rows = []
    for k, t in enumerate(trials):
        rec = {
            "trial": t["trial"],
            "query": t["query"],
            "pcd": t.get("pcd"),
            "gt_pose": t["gt_pose"],
        }
        for arm in ("raw", "star"):
            started = time.perf_counter()
            res = service.run_inference(
                SimpleNamespace(
                    query_image_filename=t["query"] + ".jpg",
                    dataset_path=ds,
                    rerank=(arm == "star"),
                )
            )
            wall = time.perf_counter() - started
            if not res:
                rec[arm] = None
                continue
            best = res["best_match_image"]
            p7 = pose7_of_image(best, pose_cache)
            rec[arm] = {
                "match": os.path.basename(best),
                "pose7": p7,
                "score": res.get("score"),
                "rank_in_raw": res.get("rank"),
                "err_vs_query_pose": res.get("distance_error"),
                "latency_s": round(wall, 3),
            }
            if arm == "raw":
                rec["raw_vpr_errors"] = res.get("raw_vpr_errors")
        g = np.array(t["gt_pose"][:3])
        for arm in ("raw", "star"):
            if rec.get(arm) and rec[arm]["pose7"]:
                rec[arm]["err_m"] = float(
                    np.linalg.norm(np.array(rec[arm]["pose7"][:3])[:2] - g[:2])
                )
        s = rec.get("star") or {}
        r = rec.get("raw") or {}
        print(
            f"  [{k + 1}/{len(trials)}] {t['query']}  raw {r.get('err_m', float('nan')):6.2f} m"
            f"   star {s.get('err_m', float('nan')):6.2f} m"
            f"   ({s.get('latency_s', 0):.1f} s)",
            flush=True,
        )
        rows.append(rec)

    out = os.path.join(a.out, "star_poses.json")
    json.dump(rows, open(out, "w", encoding="utf-8"), indent=1)
    ok = lambda arm: sum(
        1 for r in rows if r.get(arm) and r[arm].get("err_m", 1e9) <= 5.0
    )
    print(f"\n=== 检索/重排(5 m 判据) ===")
    print(f"  raw  {ok('raw')}/{len(rows)}")
    print(f"  star {ok('star')}/{len(rows)}")
    print(f"写到 {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
