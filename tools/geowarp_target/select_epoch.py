#!/usr/bin/env python


import argparse
import glob
import json
import os
import shutil
import socket
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
MAPS = [
    "RobotCar",
    "BusinessCampus",
    "Neighborhood",
    "OldTown",
    "NCLT",
    "Boreas",
    "CustomCampus",
]


def duty_pause(duty_file, elapsed):

    try:
        with open(duty_file, encoding="utf-8") as handle:
            duty = float(handle.read().strip() or 1.0)
    except (OSError, ValueError):
        duty = 1.0
    if 0.0 < duty < 1.0:
        time.sleep(elapsed * (1.0 / duty - 1.0))


def val_r1(score_candidates, limit=None, duty_file=None):
    per_map = {}
    for scene in MAPS:
        path = os.path.join(ROOT, "assets", "pairvpr_target", f"places_{scene}.json")
        with open(path, encoding="utf-8") as handle:
            val = json.load(handle)["val"][:limit]
        hit = 0
        for v in val:
            tick = time.monotonic()
            scores = score_candidates(
                os.path.join(ROOT, v["query"]),
                [os.path.join(ROOT, p) for p in v["cands"]],
            )
            best = max(range(len(scores)), key=lambda i: (scores[i], -i))
            hit += v["err"][best] <= 5.0
            if duty_file:
                duty_pause(duty_file, time.monotonic() - tick)
        per_map[scene] = round(100.0 * hit / max(1, len(val)), 1)
    return sum(per_map.values()) / len(per_map), per_map


def epoch_of(ckpt):

    return int(os.path.basename(ckpt).split("_")[-1].split(".")[0]) + 1


def write(path, log):
    log.sort(key=lambda r: r["epoch"])
    best = max(log, key=lambda r: (r["val"], -r["epoch"]))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"best": best, "log": log}, handle, indent=1)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run", required=True, help="GeoWarp run folder (has checkpoints/)"
    )
    ap.add_argument("--every", type=int, default=10, help="evaluate every N-th epoch")
    ap.add_argument(
        "--last-epoch",
        type=int,
        default=100,
        help="the run's final epoch (official default), always a candidate",
    )
    ap.add_argument(
        "--merge",
        action="store_true",
        help="keep the epochs already in --out and evaluate only the missing ones",
    )
    ap.add_argument(
        "--duty-file",
        default="/tmp/spare_geowarp_duty",
        help="fraction of wall time to compute (as for training); absent = full speed",
    )
    ap.add_argument("--limit", type=int, default=None, help="smoke tests only")
    ap.add_argument(
        "--out",
        default=os.path.join(ROOT, "assets", "geowarp_target", "selection.json"),
    )
    a = ap.parse_args()

    from run_geowarp_rerank import build_scorer

    log = []
    if a.merge and os.path.isfile(a.out):
        with open(a.out, encoding="utf-8") as handle:
            log = json.load(handle)["log"]
    done = {r["epoch"] for r in log}
    ckpts = sorted(
        glob.glob(os.path.join(a.run, "checkpoints", "homography_regression_*.torch"))
    )
    picked = [
        c
        for c in ckpts
        if (epoch_of(c) % a.every == 0 or epoch_of(c) == a.last_epoch)
        and epoch_of(c) not in done
    ]
    host = socket.gethostname()
    if 0 not in done:
        _e, score, _c = build_scorer("resnet50", "netvlad", None, 640)
        v0, m0 = val_r1(score, a.limit, a.duty_file)
        log.append(
            {"epoch": 0, "ckpt": "released", "val": v0, "per_map": m0, "host": host}
        )
        write(a.out, log)
        print(
            f"[geowarp-target] epoch 0 (released) val R@1@5m {v0:.2f} {m0}", flush=True
        )
    for ckpt in picked:
        epoch = epoch_of(ckpt)
        _e, score, _c = build_scorer("resnet50", "netvlad", ckpt, 640)
        v, m = val_r1(score, a.limit, a.duty_file)
        log.append(
            {
                "epoch": epoch,
                "ckpt": os.path.relpath(ckpt, ROOT),
                "val": v,
                "per_map": m,
                "host": host,
            }
        )
        write(a.out, log)
        print(f"[geowarp-target] epoch {epoch} val R@1@5m {v:.2f} {m}", flush=True)
    best = write(a.out, log)
    if a.last_epoch not in {r["epoch"] for r in log}:
        print(f"[geowarp-target] epoch {a.last_epoch} not evaluated yet")
    if best["epoch"] > 0 and not os.path.isfile(os.path.join(ROOT, best["ckpt"])):
        print(f"[geowarp-target] best epoch {best['epoch']} is not on this machine")
    elif best["epoch"] > 0:
        dst = os.path.join(os.path.dirname(a.out), "homography_regression_best.torch")
        shutil.copy2(os.path.join(ROOT, best["ckpt"]), dst)
        print(
            f"[geowarp-target] best epoch {best['epoch']} (val {best['val']:.2f}) -> {dst}"
        )
    else:
        print(f"[geowarp-target] released weights are best (val {best['val']:.2f})")


if __name__ == "__main__":
    main()
