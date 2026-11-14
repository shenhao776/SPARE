#!/usr/bin/env python


import argparse
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--poses",
        default=os.path.join(ROOT, "realworld", "reloc_results", "star_poses.json"),
    )
    ap.add_argument(
        "--out", default=os.path.join(ROOT, "realworld", "reloc_results", "jobs.txt")
    )
    ap.add_argument(
        "--arms", default="raw:180,star:180,star:60", help="逗号分隔的 臂:偏航半角"
    )
    a = ap.parse_args()
    rows = json.load(open(a.poses, encoding="utf-8"))
    specs = [s.split(":") for s in a.arms.split(",")]
    n = 0
    with open(a.out, "w", encoding="utf-8") as f:
        for arm, half in specs:
            for r in rows:
                d = r.get(arm)
                if not d or not d.get("pose7") or not r.get("pcd"):
                    print(f"  ⚠ trial {r['trial']} 的 {arm} 臂没位姿,跳过")
                    continue
                s = d["pose7"]
                g = r["gt_pose"]
                f.write(
                    f"{r['trial']} {arm}{half} {r['pcd']} "
                    + " ".join(f"{x:.6f}" for x in s)
                    + " "
                    + " ".join(f"{x:.6f}" for x in g)
                    + f" {half}\n"
                )
                n += 1
    print(f"[jobs] {n} 行 -> {os.path.relpath(a.out, ROOT)}")


if __name__ == "__main__":
    main()
