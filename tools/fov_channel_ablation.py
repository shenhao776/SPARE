#!/usr/bin/env python

import csv
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
PY = sys.executable
COLUMNS = [
    ("0.3-0.4", "robotcar"),
    ("0.4-0.5", "robotcarca4050"),
    ("0.5-0.6", "robotcarca5060"),
    ("0.7-0.8", "robotcarca7080"),
    ("1.0", "robotcarcfull"),
]
ARMS = [
    ("universal", "UnionMaps", {}),
    ("universal w/o zoom", "UnionMaps", {"ABLATION_USE_ZOOM": "False"}),
    ("mixed-crop", "UnionMapsMix", {}),
]


THRESHOLD_M = 5.0


def scene_threshold_m(stem):
    import yaml

    with open(
        os.path.join(ROOT, "configs", "datasets", f"{stem}.yaml"), encoding="utf-8"
    ) as fh:
        return float(yaml.safe_load(fh)["thresholds"]["test_m"])


def ours_pct(text):
    clean = re.sub(r"\x1b\[[0-9;]*m", "", text)
    block = clean.split(f"TEST THRESHOLD = {THRESHOLD_M}m")[-1]
    hits = re.findall(r"\| Ours \(arch[^|]*\|\s*\d+/\d+\s*\|\s*([\d.]+)%", block)
    if not hits:
        raise SystemExit(f"no 'Ours' row at {THRESHOLD_M:g} m:\n" + clean[-2000:])
    return float(hits[0])


def main():
    import argparse

    from core.paper_config import load_paper_config

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--stem",
        default="robotcar",
        help="基座场景 stem(robotcar / msls / pitts30k);裁剪变体为 <stem>c<tag>",
    )
    ap.add_argument("--arms", default=None, help="只跑这些臂(逗号分隔的行名),默认全部")
    a = ap.parse_args()
    global THRESHOLD_M
    THRESHOLD_M = scene_threshold_m(a.stem)
    columns = [(col, stem.replace("robotcar", a.stem, 1)) for col, stem in COLUMNS]
    arms = ARMS if not a.arms else [x for x in ARMS if x[0] in a.arms.split(",")]
    seeds = [int(s) for s in load_paper_config()["seeds"]]
    rows = []
    for arm, wscene, over in arms:
        cells = []
        for col, stem in columns:
            vals = []
            for seed in seeds:
                env = dict(
                    os.environ,
                    DATASET=stem,
                    ENABLE_SCENE_MODELS_CACHE="True",
                    STAR_WEIGHT_SCENE=wscene,
                    TRAIN_SEED=str(seed),
                    **over,
                )
                out = subprocess.run(
                    [PY, "tools/fast_offline_eval.py"],
                    cwd=ROOT,
                    env=env,
                    capture_output=True,
                    text=True,
                )
                if out.returncode:
                    raise SystemExit(
                        f"{arm} {col} seed {seed}:\n"
                        + out.stdout[-1500:]
                        + out.stderr[-1500:]
                    )
                vals.append(ours_pct(out.stdout))
            mean = sum(vals) / len(vals)
            std = (sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
            cells.append(f"{mean:.1f}±{std:.1f}")
            print(f"[fov-abl] {arm:<22} {col:<8} {mean:.1f}±{std:.1f}", flush=True)
        rows.append([arm] + cells)
    suffix = "" if a.stem == "robotcar" else f"_{a.stem}"
    out_csv = os.path.join(
        ROOT, "results", "tables", f"fov_channel_ablation{suffix}.csv"
    )
    if a.arms and os.path.isfile(out_csv):
        with open(out_csv, encoding="utf-8") as fh:
            old = [r for r in csv.reader(fh)][1:]
        done = {r[0] for r in rows}
        rows = [r for r in old if r[0] not in done] + rows
        rows.sort(
            key=lambda r: (
                [x[0] for x in ARMS].index(r[0]) if r[0] in [x[0] for x in ARMS] else 99
            )
        )
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["head"] + [c for c, _ in columns])
        w.writerows(rows)
    print(open(out_csv, encoding="utf-8").read())


if __name__ == "__main__":
    main()
