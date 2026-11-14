#!/usr/bin/env python


import argparse
import csv
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEPLOY_YAW = "60"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(ROOT, "realworld", "reloc_results"))
    ap.add_argument("--basin", type=float, default=10.0, help="NDT 的平移搜索半径")
    ap.add_argument("--hit", type=float, default=1.0, help="重定位算成功的距离")
    ap.add_argument(
        "--paired-csv",
        default=None,
        help="逐试验配对的 2x2 计数(both / only_first / only_second / neither)与精确 "
        "McNemar 双侧 p,SPARE vs raw、SPARE vs FoL,60/180 两档各一行。生成的,不手打。",
    )
    ap.add_argument(
        "--csv",
        default=None,
        help="同时把这些数写成 CSV(论文的数值凭据)。**生成的,不是手打的** —— "
        "手打一份就等于给同一个数留了第二份拷贝,而两份不一致时谁都不会报错",
    )
    a = ap.parse_args()

    poses = json.load(open(os.path.join(a.dir, "star_poses.json"), encoding="utf-8"))
    n = len(poses)
    print(f"试验数 {n}\n")

    arms = [("raw", "base retriever")]
    if any("fol" in r for r in poses):
        arms.append(("fol", "FoL (strongest baseline)"))
    arms.append(("star", "+SPARE"))
    print("检索 / 重排 —— Top-1 位置误差")
    for arm, name in arms:
        e = np.array([r[arm]["err_m"] for r in poses])
        cells = "/".join(f"{100 * (e <= t).mean():.1f}" for t in (5, 10, 15))
        print(f"  {name:<15} 5/10/15 m: {cells} %" f"   (中位 {np.median(e):.1f} m)")
    lat = {
        arm: np.array([r[arm]["latency_s"] for r in poses]) for arm in ("raw", "star")
    }
    print(
        f"  一次请求 {lat['star'].mean():.1f}+-{lat['star'].std():.1f} s"
        f"(含逐 query 调试产物;不含时另测)\n"
    )

    import glob

    rows = []
    for f in sorted(glob.glob(os.path.join(a.dir, "ndt*.csv"))):
        if os.path.basename(f).startswith("ndt_iters"):
            continue
        rows += list(csv.DictReader(open(f, encoding="utf-8")))
    by = {}
    for r in rows:
        by.setdefault(r["arm"], []).append(r)
    print(f"端到端重定位(收敛且 < {a.hit:g} m)")
    for arm in sorted(by):
        rs = by[arm]
        err = np.array([float(r["err_m"]) for r in rs])
        seed = np.array([float(r["seed_err_m"]) for r in rs])
        sec = np.array([float(r["sec"]) for r in rs])
        ok = err < a.hit
        near = seed <= a.basin
        print(
            f"  {arm:<9} {ok.sum():2d}/{len(rs)}   "
            f"初值 <= {a.basin:g} m: {(ok & near).sum()}/{near.sum()}   "
            f"> {a.basin:g} m: {(ok & ~near).sum()}/{(~near).sum()}   "
            f"耗时 {sec.mean():.1f}+-{sec.std():.1f} s"
        )
    print("\n重排的作用 = 把初值搬进收敛域:", end=" ")
    near_n = {}
    for arm, _ in arms:
        rs = by.get(f"{arm}{DEPLOY_YAW}")
        if rs is None:
            e = np.array([r[arm]["err_m"] for r in poses])
        else:
            e = np.array([float(r["seed_err_m"]) for r in rs])
        near_n[arm] = int((e <= a.basin).sum())
        print(f"{arm} {near_n[arm]}/{n}", end="   ")
    print()

    if a.csv:

        def triple(arm):
            e = np.array([r[arm]["err_m"] for r in poses])
            return "/".join(f"{100 * (e <= t).mean():.1f}" for t in (5, 10, 15))

        def reloc(arm_csv):
            rs = by[arm_csv]
            err = np.array([float(r["err_m"]) for r in rs])
            seed = np.array([float(r["seed_err_m"]) for r in rs])
            sec = np.array([float(r["sec"]) for r in rs])
            ok, near = err < a.hit, seed <= a.basin
            return {
                "reloc_success": f"{ok.sum()}/{len(rs)}",
                "reloc_given_near": f"{(ok & near).sum()}/{near.sum()}",
                "reloc_given_far": f"{(ok & ~near).sum()}/{(~near).sum()}",
                "reloc_seconds": f"{sec.mean():.1f}",
            }

        cols = {}
        for name, parm in (("base", "raw"), ("fol", "fol"), ("star", "star")):
            if parm != "fol" or any("fol" in r for r in poses):
                for suf in ("60", "180"):
                    key = name if suf == "60" else f"{name}_yaw180"
                    if f"{parm}{suf}" in by:
                        cols[key] = (parm, f"{parm}{suf}")
        rows = {"retrieval_5_10_15": {}, "seed_within_basin": {}}
        for name, (parm, narm) in cols.items():
            rows["retrieval_5_10_15"][name] = triple(parm)
            rows["seed_within_basin"][name] = f"{near_n[parm]}/{n}"
            for k, v in reloc(narm).items():
                rows.setdefault(k, {})[name] = v
        import csv as _csv

        with open(a.csv, "w", encoding="utf-8", newline="") as fh:
            w = _csv.writer(fh)
            w.writerow(["metric"] + list(cols))
            for k, d in rows.items():
                w.writerow([k] + [d[c] for c in cols])
        print(f"\n写出 {a.csv}")

    if a.paired_csv:
        from math import comb

        def success_by_trial(arm_csv):
            return {
                int(r["trial"]): (
                    int(float(r["converged"])) == 1 and float(r["err_m"]) < a.hit
                )
                for r in by.get(arm_csv, [])
            }

        def mcnemar_exact(b, c):
            n_disc = b + c
            if n_disc == 0:
                return 1.0
            k = min(b, c)
            tail = sum(comb(n_disc, i) for i in range(k + 1)) / 2**n_disc
            return min(1.0, 2 * tail)

        import csv as _csv

        pairs = []
        for suf in ("60", "180"):
            star = success_by_trial(f"star{suf}")
            if not star:
                continue
            for other_name, other_csv in (("raw", f"raw{suf}"), ("fol", f"fol{suf}")):
                other = success_by_trial(other_csv)
                if not other:
                    continue
                trials = sorted(set(star) & set(other))
                both = sum(star[t] and other[t] for t in trials)
                only_star = sum(star[t] and not other[t] for t in trials)
                only_other = sum(other[t] and not star[t] for t in trials)
                neither = len(trials) - both - only_star - only_other
                pairs.append(
                    [
                        f"star_vs_{other_name}_yaw{suf}",
                        len(trials),
                        both,
                        only_star,
                        only_other,
                        neither,
                        f"{mcnemar_exact(only_star, only_other):.4f}",
                    ]
                )
        with open(a.paired_csv, "w", encoding="utf-8", newline="") as fh:
            w = _csv.writer(fh)
            w.writerow(
                [
                    "pair",
                    "n",
                    "both",
                    "only_spare",
                    "only_other",
                    "neither",
                    "mcnemar_exact_p",
                ]
            )
            w.writerows(pairs)
        print(f"\n写出 {a.paired_csv}")
        for row in pairs:
            print("  " + " ".join(str(x) for x in row))


if __name__ == "__main__":
    main()
