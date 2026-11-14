#!/usr/bin/env python


import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "realworld", "reloc_results")
QDIR = os.path.join(
    ROOT,
    "datasets",
    "CustomCampus",
    "queries_test",
    "recording_full",
    "undistorted_images",
)
DDIR = os.path.join(
    ROOT,
    "datasets",
    "CustomCampus",
    "database",
    "recording_mapping",
    "undistorted_images",
)


def fmt_m(v):
    return f"{v:.0f} m" if v >= 10 else (f"{v:.1f} m" if v >= 1 else f"{v:.2f} m")


def load_ndt(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[(int(r["trial"]), r["arm"])] = r
    return out


def main():
    global RES
    ap = argparse.ArgumentParser()
    ap.add_argument("--row-a", type=int, required=True, help="冷启动那一行用哪次试验")
    ap.add_argument(
        "--row-b",
        type=int,
        default=None,
        help="照片导航那一行用哪次冷启动试验;B 行是手持拍照的现场演示,不是冷启动 trial,所以默认不生成",
    )
    ap.add_argument("--out", default=os.path.join(ROOT, "realworld", "fig5_assets"))
    ap.add_argument(
        "--map",
        default=os.path.expanduser(
            "~/datasets/my_map_data/PCD/map_final_visualization.pcd"
        ),
    )
    ap.add_argument(
        "--aspect",
        type=float,
        default=252.0 / 168.0,
        help="俯视图宽高比,必须等于排版脚本的 COL_W/IMG_H,否则会被裁一圈",
    )
    ap.add_argument(
        "--rotate-deg",
        type=float,
        default=0.0,
        help="俯视图整体旋转角(两格相同),把真值从角标底下挪开",
    )
    ap.add_argument(
        "--reloc-arm",
        default="star180",
        help="下格用哪个偏航搜索设置的 NDT 结果(默认部署默认值 180 度)",
    )
    ap.add_argument(
        "--res",
        default=RES,
        help="读哪个跑批目录的 star_poses.json / ndt.csv(默认 realworld/reloc_results;"
        "通用 head 的结果在 realworld/reloc_results_union)",
    )
    a = ap.parse_args()
    RES = a.res
    os.makedirs(a.out, exist_ok=True)

    poses = {
        r["trial"]: r
        for r in json.load(open(os.path.join(RES, "star_poses.json"), encoding="utf-8"))
    }
    ndt = load_ndt(os.path.join(RES, "ndt.csv"))
    raw_arm = a.reloc_arm.replace("star", "raw")
    labels, note = {}, []

    rows = [("a", a.row_a)] + ([("b", a.row_b)] if a.row_b is not None else [])
    for row, trial in rows:
        r = poses[trial]
        shutil.copy(
            os.path.join(QDIR, r["query"] + ".jpg"),
            os.path.join(a.out, f"{row}_query.jpg"),
        )
        for arm in ("raw", "star"):
            shutil.copy(
                os.path.join(DDIR, r[arm]["match"]),
                os.path.join(a.out, f"{row}_{arm}_frame.jpg"),
            )
            e = r[arm]["err_m"]
            labels[f"{row}_{arm}_frame"] = {
                "text": fmt_m(e),
                "mark": "ok" if e <= 5.0 else "bad",
                "color": "green" if e <= 5.0 else "red",
            }
        note.append(
            f"row {row.upper()} = trial {trial} ({r['query']}): "
            f"raw {r['raw']['err_m']:.2f} m, star {r['star']['err_m']:.2f} m "
            f"(raw rank {r['star']['rank_in_raw']})"
        )
        print("  " + note[-1])

    ra = poses[a.row_a]
    gx, gy = ra["gt_pose"][0], ra["gt_pose"][1]
    try:
        d_raw = ndt[(a.row_a, raw_arm)]
        d_star = ndt[(a.row_a, a.reloc_arm)]
    except KeyError as exc:
        raise SystemExit(f"ndt.csv 里没有 trial {a.row_a} 的 {exc} —— 先把 NDT 跑完")
    fr = (float(d_raw["fx"]), float(d_raw["fy"]))
    fs = (float(d_star["fx"]), float(d_star["fy"]))
    spread = max(math.hypot(fr[0] - gx, fr[1] - gy), math.hypot(fs[0] - gx, fs[1] - gy))
    far = fr if math.hypot(fr[0] - gx, fr[1] - gy) >= spread - 1e-6 else fs
    cx, cy = (gx + far[0]) / 2, (gy + far[1]) / 2
    radius = max(18.0, 0.62 * spread + 10.0)
    print(f"  [reloc] 同框:中心 ({cx:.1f}, {cy:.1f}) 半径 {radius:.1f} m")

    for slot, d in ((f"a_raw_reloc", d_raw), (f"a_star_reloc", d_star)):
        out_png = os.path.join(a.out, slot + ".pdf")
        subprocess.run(
            [
                sys.executable,
                os.path.join(ROOT, "realworld", "render_reloc_overlay.py"),
                "--map",
                a.map,
                "--scan",
                os.path.join(ROOT, "realworld", "reloc_frames", ra["pcd"]),
                "--pose",
                d["fx"],
                d["fy"],
                d["fz"],
                d["fqx"],
                d["fqy"],
                d["fqz"],
                d["fqw"],
                "--gt",
                f"{gx}",
                f"{gy}",
                "--center",
                f"{cx}",
                f"{cy}",
                "--radius",
                f"{radius}",
                "--slice-z",
                f"{ra['gt_pose'][2]}",
                "--aspect",
                f"{a.aspect}",
                "--rotate-deg",
                f"{a.rotate_deg}",
                "--out",
                out_png,
            ],
            check=True,
        )
        e = float(d["err_m"])
        labels[slot] = {
            "text": fmt_m(e),
            "mark": "ok" if e < 1.0 else "bad",
            "color": "green" if e < 1.0 else "red",
        }
        note.append(
            f"  {slot}: seed {float(d['seed_err_m']):.2f} m -> "
            f"err {e:.3f} m, fitness {float(d['fitness']):.3f}, "
            f"{float(d['sec']):.1f} s"
        )

    with open(os.path.join(a.out, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=1, ensure_ascii=False)
    with open(os.path.join(a.out, "PROVENANCE.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(note) + "\n")
        f.write(
            "row B:手持拍照的照片导航演示,不是冷启动 trial;真值 = 拍照人实地位置 "
            "(RViz /clicked_point 18.2094, 80.8282);raw 匹配帧 49.19 m / SPARE 匹配帧 2.95 m / "
            "raw 停车点 49.03 m / SPARE 停车点 3.22 m(停车点 = 两个存档 bag 的 /pcl_pose 终点),用 5 m head。"
            "b_raw_robot / b_star_robot 为现场补拍,素材见 figs/robot_demo/。\n"
        )
    print(f"[out] {a.out}  —— 只差 b_raw_robot / b_star_robot 两张现场照片")


if __name__ == "__main__":
    main()
