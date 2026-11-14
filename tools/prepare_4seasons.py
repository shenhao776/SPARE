#!/usr/bin/env python


import argparse
import bisect
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile

RAW_ROOT = os.path.expanduser("~/datasets/4seasons")
TOL_NS = 2_000_000


def find_zip(route, recording, kind):
    for sub in (
        "",
        "database",
        "test",
        "raw",
    ):
        p = os.path.join(RAW_ROOT, route, sub, f"recording_{recording}_{kind}.zip")
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        f"{route}/recording_{recording}_{kind}.zip 不在 database/、test/ 或 raw/ 下"
    )


def keep_set(result_txt, frame_ts):

    rows = sorted(
        round(float(l.split()[0]) * 1e9)
        for l in open(result_txt, encoding="utf-8")
        if l.strip() and not l.startswith("#")
    )
    frames = sorted(frame_ts)
    used = [False] * len(frames)
    kept = set()
    for r in rows:
        i = bisect.bisect(frames, r)
        best = None
        for j in (i - 1, i):
            if 0 <= j < len(frames) and not used[j] and abs(frames[j] - r) <= TOL_NS:
                if best is None or abs(frames[j] - r) < abs(frames[best] - r):
                    best = j
        if best is not None:
            used[best] = True
            kept.add(frames[best])
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--route", required=True)
    ap.add_argument("--recording", required=True, help="如 2021-01-07_13-12-23")
    ap.add_argument(
        "--role", required=True, help="recording_mapping / recording_1 / recording_2"
    )
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--validate",
        default=None,
        help="与该现有 recording 目录比对文件集合与位姿文件后报告",
    )
    a = ap.parse_args()

    dst = os.path.join(a.out, a.route, a.role)
    if os.path.exists(dst) and os.listdir(dst):
        print(f"[skip] {dst} 已存在且非空(存在性判据)")
        return
    poses_zip = find_zip(a.route, a.recording, "reference_poses")
    images_zip = find_zip(a.route, a.recording, "stereo_images_undistorted")
    os.makedirs(dst, exist_ok=True)

    inner = f"recording_{a.recording}"
    with zipfile.ZipFile(poses_zip) as z:
        for name in z.namelist():
            base = os.path.basename(name)
            if base in ("GNSSPoses.txt", "Transformations.txt", "result.txt"):
                with z.open(name) as src, open(os.path.join(dst, base), "wb") as out:
                    shutil.copyfileobj(src, out)
    for need in ("GNSSPoses.txt", "Transformations.txt", "result.txt"):
        if not os.path.isfile(os.path.join(dst, need)):
            raise RuntimeError(f"{poses_zip} 里没有 {need}")

    cam_dir = os.path.join(dst, "undistorted_images", "cam0")
    os.makedirs(cam_dir, exist_ok=True)
    with zipfile.ZipFile(images_zip) as z:
        cam0 = [
            n
            for n in z.namelist()
            if "/undistorted_images/cam0/" in n and n.endswith(".png")
        ]
        ts_of = {n: int(os.path.basename(n)[:-4]) for n in cam0}
        kept = keep_set(os.path.join(dst, "result.txt"), ts_of.values())
        n_write = 0
        for n in cam0:
            if ts_of[n] in kept:
                with z.open(n) as src, open(
                    os.path.join(cam_dir, os.path.basename(n)), "wb"
                ) as out:
                    shutil.copyfileobj(src, out)
                n_write += 1
    print(
        f"[done] {a.route}/{a.role} <- recording_{a.recording}: "
        f"zip {len(cam0)} 帧 -> 保留 {n_write}"
    )

    if a.validate:
        ref = set(os.listdir(os.path.join(a.validate, "undistorted_images", "cam0")))
        new = set(os.listdir(cam_dir))
        print(
            f"[validate] 现有 {len(ref)} vs 重建 {len(new)}:"
            f"交集 {len(ref & new)} 仅现有 {len(ref - new)} 仅重建 {len(new - ref)}"
        )


if __name__ == "__main__":
    main()
