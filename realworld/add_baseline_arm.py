#!/usr/bin/env python


import argparse
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "realworld", "reloc_results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="fol")
    ap.add_argument("--scene", default="CustomCampus")
    ap.add_argument("--pool", default="MegaLoc_8448")
    ap.add_argument("--arm", default=None, help="写进 star_poses.json 的臂名")
    ap.add_argument("--poses", default=os.path.join(RES, "star_poses.json"))
    a = ap.parse_args()
    arm = a.arm or a.baseline

    cache = os.path.join(
        ROOT, "assets", a.pool, f"test_trajectories_{a.scene}_no_roi.json"
    )
    res = os.path.join(
        ROOT,
        "assets",
        "baseline_results",
        f"{a.baseline}_rerank_{a.scene}__pool-{a.pool}.json",
    )
    for p in (cache, res):
        if not os.path.isfile(p):
            raise SystemExit(f"找不到 {p}")

    rows = json.load(open(cache, encoding="utf-8"))
    order, by_query = [], {}
    for r in rows:
        q = r["query"]
        if q not in by_query:
            by_query[q] = {}
            order.append(q)
        by_query[q][int(r["rank"])] = r["candidate_name"]
    idx = {q: i for i, q in enumerate(order)}

    b = json.load(open(res, encoding="utf-8"))
    ranks = b["selected_ranks"]
    new_err = b["reranked_errors_m"]
    if len(ranks) != len(order):
        raise SystemExit(f"基线有 {len(ranks)} 条,缓存有 {len(order)} 个 query")

    from run_reloc_trials import pose7_of_image

    ddir = os.path.join(ROOT, "datasets", a.scene, "database")
    seq = None
    for dp, _, fs in os.walk(ddir):
        if any(f.endswith((".jpg", ".png")) for f in fs):
            seq = dp
            break
    if seq is None:
        raise SystemExit(f"{ddir} 下找不到数据库图像")

    trials = json.load(open(a.poses, encoding="utf-8"))
    cache_pose, n = {}, 0
    for t in trials:
        q = t["query"]
        if q not in idx:
            print(f"  ⚠ trial {t['trial']} 的 query 不在缓存里,跳过")
            continue
        i = idx[q]
        if by_query[q].get(1) != t["raw"]["match"]:
            raise SystemExit(
                f"trial {t['trial']} ({q}) 索引对不上:缓存 rank-1 是 "
                f"{by_query[q].get(1)},在线 raw 选的是 {t['raw']['match']} —— "
                f"顺序假设不成立,拒绝继续。"
            )
        name = by_query[q].get(int(ranks[i]))
        if name is None:
            print(f"  ⚠ trial {t['trial']}: 缓存里没有 rank {ranks[i]}")
            continue
        p7 = pose7_of_image(os.path.join(seq, name), cache_pose)
        t[arm] = {
            "match": name,
            "pose7": p7,
            "rank_in_raw": int(ranks[i]),
            "err_m": float(new_err[i]),
        }
        n += 1

    json.dump(trials, open(a.poses, "w", encoding="utf-8"), indent=1)
    e = np.array([t[arm]["err_m"] for t in trials if arm in t])
    print(f"[{arm}] 写入 {n}/{len(trials)} 条 -> {os.path.relpath(a.poses, ROOT)}")
    print(
        "  Top-1 recall  "
        + "  ".join(f"@{th}m {int((e <= th).sum())}/{len(e)}" for th in (5, 10, 15))
    )


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.join(ROOT, "realworld"))
    main()
