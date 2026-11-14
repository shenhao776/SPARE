#!/usr/bin/env python


import argparse
import csv
import math
import os
import shutil

RAW_ROOT = os.path.expanduser("~/datasets/Boreas")
MIN_STEP_M = 1.0


def quat_xyz(roll, pitch, heading):

    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    ch, sh = math.cos(heading / 2), math.sin(heading / 2)
    qw = ch * cp * cr + sh * sp * sr
    qx = ch * cp * sr - sh * sp * cr
    qy = ch * sp * cr + sh * cp * sr
    qz = sh * cp * cr - ch * sp * sr
    return qx, qy, qz, qw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence", required=True, help="如 boreas-2021-01-15-12-17")
    ap.add_argument(
        "--role", required=True, help="recording_mapping / recording_1 / recording_2"
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--validate", default=None)
    a = ap.parse_args()

    seq_dir = None
    for d in os.listdir(RAW_ROOT):
        if d == a.sequence or d.startswith(a.sequence):
            seq_dir = os.path.join(RAW_ROOT, d)
            break
    if seq_dir is None:
        raise FileNotFoundError(f"{RAW_ROOT} 下没有 {a.sequence}*")

    dst = os.path.join(a.out, "Boreas", a.role)
    if os.path.exists(dst) and os.listdir(dst):
        print(f"[skip] {dst} 已存在且非空(存在性判据)")
        return
    os.makedirs(dst, exist_ok=True)

    cam_dir = os.path.join(seq_dir, "camera")
    frames = {int(f[:-4]) for f in os.listdir(cam_dir) if f.endswith(".png")}

    kept, last = [], None
    with open(os.path.join(seq_dir, "applanix", "camera_poses.csv")) as fh:
        for row in csv.DictReader(fh):
            ts = int(row["GPSTime"])
            if ts not in frames:
                continue
            x, y, z = (
                float(row["easting"]),
                float(row["northing"]),
                float(row["altitude"]),
            )
            if last is not None and math.dist((x, y), last[:2]) < MIN_STEP_M:
                continue
            last = (x, y)
            qx, qy, qz, qw = quat_xyz(
                float(row["roll"]), float(row["pitch"]), float(row["heading"])
            )
            kept.append((ts, x, y, z, qx, qy, qz, qw))

    with open(os.path.join(dst, "pose.txt"), "w") as out:
        for ts, x, y, z, qx, qy, qz, qw in kept:
            out.write(
                f"{ts * 1000} {x:.6f} {y:.6f} {z:.6f} "
                f"{qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}\n"
            )
    for ts, *_ in kept:
        shutil.copy2(os.path.join(cam_dir, f"{ts}.png"), os.path.join(dst, f"{ts}.png"))
    print(
        f"[done] Boreas/{a.role} <- {a.sequence}: csv 帧 {len(frames)} -> 保留 {len(kept)}"
    )

    if a.validate:
        ref = {f for f in os.listdir(a.validate) if f.endswith(".png")}
        new = {f"{ts}.png" for ts, *_ in kept}
        print(
            f"[validate] 图像集合:现有 {len(ref)} vs 重建 {len(new)}:"
            f"交集 {len(ref & new)} 仅现有 {len(ref - new)} 仅重建 {len(new - ref)}"
        )
        ref_pose = [l.split() for l in open(os.path.join(a.validate, "pose.txt"))]
        new_pose = {
            l[0]: l[1:]
            for l in (x.split() for x in open(os.path.join(dst, "pose.txt")))
        }
        worst = 0.0
        n_cmp = 0
        for rp in ref_pose:
            np_ = new_pose.get(rp[0])
            if np_ is None:
                continue
            n_cmp += 1
            worst = max(
                worst, max(abs(float(u) - float(v)) for u, v in zip(rp[1:], np_))
            )
        print(
            f"[validate] pose.txt 可比行 {n_cmp}/{len(ref_pose)},逐字段最大偏差 {worst:.2e}"
        )


if __name__ == "__main__":
    main()
