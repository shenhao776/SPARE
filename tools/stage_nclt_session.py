#!/usr/bin/env python


import argparse
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SRC = os.path.expanduser("~/datasets/NCLTDataset")


def to_tum(csv_path, out_path):

    import csv
    import math

    n = 0
    with open(csv_path, newline="", encoding="utf-8") as fi, open(
        out_path + ".partial", "w", encoding="utf-8"
    ) as fo:
        for row in csv.reader(fi):
            if len(row) < 7:
                continue
            try:
                t = float(row[0]) / 1e6
                x, y, z = (float(row[1]), float(row[2]), float(row[3]))
                r, p, h = (float(row[4]), float(row[5]), float(row[6]))
            except ValueError:
                continue
            if any(v != v for v in (x, y, z, r, p, h)):
                continue
            cy, sy = math.cos(h * 0.5), math.sin(h * 0.5)
            cp, sp = math.cos(p * 0.5), math.sin(p * 0.5)
            cr, sr = math.cos(r * 0.5), math.sin(r * 0.5)
            qw = cr * cp * cy + sr * sp * sy
            qx = sr * cp * cy - cr * sp * sy
            qy = cr * sp * cy + sr * cp * sy
            qz = cr * cp * sy - sr * sp * cy
            fo.write(
                f"{t:.9f} {x:.9f} {y:.9f} {z:.9f} "
                f"{qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}\n"
            )
            n += 1
    os.replace(out_path + ".partial", out_path)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="e.g. 2012-05-26")
    ap.add_argument("--cam", type=int, default=5)
    ap.add_argument("--out", required=True, help="recording dir to create")
    ap.add_argument(
        "--interval",
        type=int,
        default=4,
        help="keep every Nth frame; 4 reproduces the measured 0.8 s",
    )
    ap.add_argument("--scratch", default=os.environ.get("SCRATCH", "/tmp/nclt_stage"))
    ap.add_argument(
        "--keep-tiff",
        action="store_true",
        help="do not delete the extracted camera afterwards",
    )
    a = ap.parse_args()

    from PIL import Image

    tar = os.path.join(SRC, f"{a.date}_lb3.tar.gz")
    csv_path = os.path.join(SRC, f"groundtruth_{a.date}.csv")
    for p in (tar, csv_path):
        if not os.path.isfile(p):
            raise SystemExit(f"missing: {p}")

    raw = os.path.join(a.scratch, a.date)
    member = f"{a.date}/lb3/Cam{a.cam}"
    cam_dir = os.path.join(raw, member)
    if not os.path.isdir(cam_dir):
        os.makedirs(raw, exist_ok=True)
        print(
            f"[1/3] 解压 {member} <- {os.path.basename(tar)} "
            f"(gzip 只能顺序解,一趟约 20-30 分钟)",
            flush=True,
        )
        subprocess.run(["tar", "-xzf", tar, "-C", raw, member], check=True)
    files = sorted(f for f in os.listdir(cam_dir) if f.endswith(".tiff"))
    print(f"[1/3] Cam{a.cam} 共 {len(files)} 帧", flush=True)

    keep = files[:: a.interval]
    img_out = os.path.join(a.out, "undistorted_images")
    os.makedirs(img_out, exist_ok=True)
    print(f"[2/3] 每 {a.interval} 帧取一 -> {len(keep)} 帧,转 90 度顺时针", flush=True)
    done = 0
    for f in keep:
        dst = os.path.join(img_out, f)
        if os.path.isfile(dst):
            done += 1
            continue
        im = Image.open(os.path.join(cam_dir, f)).convert("RGB")
        im.transpose(Image.ROTATE_270).save(
            dst + ".partial", format="TIFF", compression="jpeg", quality=75
        )
        os.replace(dst + ".partial", dst)
        done += 1
        if done % 500 == 0:
            print(f"      {done}/{len(keep)}", flush=True)
    first = Image.open(os.path.join(img_out, keep[0]))
    print(
        f"[2/3] 完成 {done} 帧,尺寸 {first.size}"
        f"{'  ⚠️ 应为 (1232, 1616)' if first.size != (1232, 1616) else ''}",
        flush=True,
    )

    n = to_tum(csv_path, os.path.join(a.out, "pose.txt"))
    print(f"[3/3] pose.txt {n} 行(全量真值,按时间戳最近邻查,不是每帧一行)")

    if not a.keep_tiff:
        shutil.rmtree(cam_dir, ignore_errors=True)
        print(f"      已删中间解压件 {cam_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
