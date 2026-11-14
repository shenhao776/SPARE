#!/usr/bin/env python


import argparse
import csv
import json
import os
from collections import OrderedDict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POOL = "MegaLoc_8448"
SCENES = [
    "RobotCar",
    "BusinessCampus",
    "Neighborhood",
    "OldTown",
    "NCLT",
    "Boreas",
    "CustomCampus",
    "Pitts30k",
    "MSLS",
]
FLAT = {"Pitts30k", "MSLS"}
CUES = {
    "inliers": lambda c: c["inliers"][len(c["inliers"]) // 2],
    "dino": lambda c: c["dino_sim"][len(c["dino_sim"]) // 2],
    "vprp": lambda c: c["vprp_sim"][0],
}
VARIANTS = OrderedDict(
    [
        ("raw", []),
        ("inliers", ["inliers"]),
        ("dino+vprp", ["dino", "vprp"]),
        ("inliers+vprp", ["inliers", "vprp"]),
        ("inliers+dino+vprp", ["inliers", "dino", "vprp"]),
    ]
)


def radii(scene):
    return (10.0, 15.0, 25.0) if scene in FLAT else (5.0, 10.0, 15.0)


def shortlists(scene):
    path = os.path.join(ROOT, "assets", POOL, f"test_trajectories_{scene}_no_roi.json")
    with open(path, encoding="utf-8") as handle:
        rows = json.load(handle)
    grouped = OrderedDict()
    for row in rows:
        grouped.setdefault(row["query"], []).append(row)
    return {q: sorted(c, key=lambda r: r["rank"])[:10] for q, c in grouped.items()}


def select(cands, cues):

    total = [0] * len(cands)
    for cue in cues:
        value = [CUES[cue](c) for c in cands]
        order = sorted(range(len(cands)), key=lambda i: (-value[i], i))
        for position, i in enumerate(order):
            total[i] += position
    return min(range(len(cands)), key=lambda i: (total[i], i))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    table = []
    for scene in SCENES:
        lists = shortlists(scene)
        per_query = {}
        for name, cues in VARIANTS.items():
            picks = [select(c, cues) for c in lists.values()]
            errors = [float(c[i]["error"]) for c, i in zip(lists.values(), picks)]
            recall = [
                100.0 * sum(0 <= e <= r for e in errors) / len(errors)
                for r in radii(scene)
            ]
            table.append([scene, name, len(errors)] + [f"{v:.2f}" for v in recall])
            per_query[name] = {
                "selected_ranks": [c[i]["rank"] for c, i in zip(lists.values(), picks)],
                "reranked_errors_m": errors,
            }
            print(f"{scene:15s} {name:18s} " + "/".join(f"{v:.1f}" for v in recall))
        with open(
            os.path.join(a.out, f"rank_fusion_{scene}.json"), "w", encoding="utf-8"
        ) as handle:
            json.dump({"scene": scene, "pool": POOL, "variants": per_query}, handle)
    with open(
        os.path.join(a.out, "rank_fusion.csv"), "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["scene", "variant", "queries", "r_a", "r_b", "r_c"])
        writer.writerows(table)


if __name__ == "__main__":
    main()
