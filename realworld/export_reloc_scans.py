#!/usr/bin/env python


import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SESSION = os.path.expanduser("~/datasets/CustomCampus_raw/CustomCampus_2026-07-31_Sun")
QUERY_DIR = os.path.join(
    ROOT, "datasets", "CustomCampus", "queries_test", "recording_full"
)


def farthest_point_sample(xy, n):

    picked = [0]
    d = np.linalg.norm(xy - xy[0], axis=1)
    while len(picked) < n:
        j = int(np.argmax(d))
        picked.append(j)
        d = np.minimum(d, np.linalg.norm(xy - xy[j], axis=1))
    return picked


def pc2_to_xyz(msg):

    offs = {f.name: f.offset for f in msg.fields}
    if not {"x", "y", "z"} <= set(offs):
        return None
    n = msg.width * msg.height
    raw = np.frombuffer(msg.data, dtype=np.uint8, count=n * msg.point_step)
    raw = raw.reshape(n, msg.point_step)
    out = np.empty((n, 3), dtype=np.float32)
    for k, name in enumerate("xyz"):
        o = offs[name]
        out[:, k] = raw[:, o : o + 4].copy().view(np.float32).ravel()
    return out


def write_pcd(path, pts):
    pts = pts[np.isfinite(pts).all(axis=1)]
    with open(path, "wb") as f:
        f.write(
            (
                "# .PCD v0.7\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\n"
                "TYPE F F F\nCOUNT 1 1 1\n"
                f"WIDTH {len(pts)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n"
                f"POINTS {len(pts)}\nDATA binary\n"
            ).encode()
        )
        f.write(pts.astype(np.float32).tobytes())
    return len(pts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", default=os.path.join(ROOT, "realworld", "reloc_frames"))
    ap.add_argument("--session", default=SESSION)
    ap.add_argument(
        "--max-dt-ms",
        type=float,
        default=100.0,
        help="query 时间戳与雷达帧的最大间隔。雷达 10 Hz,超过就说明那一段缺帧",
    )
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    pose_txt = os.path.join(QUERY_DIR, "pose.txt")
    if not os.path.isfile(pose_txt):
        raise SystemExit(f"找不到 {pose_txt}")
    P = np.loadtxt(pose_txt)
    stems = [
        os.path.splitext(f)[0]
        for f in sorted(os.listdir(os.path.join(QUERY_DIR, "undistorted_images")))
    ]
    if len(stems) != len(P):
        raise SystemExit(f"图 {len(stems)} 张 vs pose {len(P)} 行,对不上")

    picks = farthest_point_sample(P[:, 1:3], min(a.n, len(P)))
    want = {}
    for rank, i in enumerate(picks):
        ts_ns = int(stems[i])
        want[ts_ns] = {
            "trial": rank,
            "query": stems[i],
            "gt_pose_img": [float(x) for x in P[i, 1:]],
        }
    print(f"[pick] {len(want)} 帧,最远点采样自 {len(P)} 帧")

    from rosbags.rosbag2 import Reader
    from rosbags.typesys import Stores, get_types_from_msg, get_typestore

    ts_store = get_typestore(Stores.ROS2_HUMBLE)
    livox = {}
    livox.update(
        get_types_from_msg(
            "uint32 offset_time\nfloat32 x\nfloat32 y\nfloat32 z\n"
            "uint8 reflectivity\nuint8 tag\nuint8 line\n",
            "livox_ros_driver2/msg/CustomPoint",
        )
    )
    livox.update(
        get_types_from_msg(
            "std_msgs/Header header\nuint64 timebase\nuint32 point_num\n"
            "uint8 lidar_id\nuint8[3] rsvd\nCustomPoint[] points\n",
            "livox_ros_driver2/msg/CustomMsg",
        )
    )
    ts_store.register(livox)

    bags = sorted(
        os.path.join(a.session, "ros2_bag", d)
        for d in os.listdir(os.path.join(a.session, "ros2_bag"))
    )
    tol = int(a.max_dt_ms * 1e6)
    done = {}
    for bag in bags:
        remaining = [t for t in want if t not in done]
        if not remaining:
            break
        try:
            with Reader(bag) as r:
                conns = {c.topic: c for c in r.connections}
                if "/livox/point" not in conns or "/pcl_pose" not in conns:
                    continue
                lo = min(remaining) - tol
                hi = max(remaining) + tol
                if r.end_time < lo or r.start_time > hi:
                    continue
                pt, pv = [], []
                for c, t, raw in r.messages(connections=[conns["/pcl_pose"]]):
                    m = ts_store.deserialize_cdr(raw, c.msgtype)
                    p = m.pose.pose
                    pt.append(t)
                    pv.append(
                        [
                            p.position.x,
                            p.position.y,
                            p.position.z,
                            p.orientation.x,
                            p.orientation.y,
                            p.orientation.z,
                            p.orientation.w,
                        ]
                    )
                if not pt:
                    continue
                pt = np.asarray(pt, dtype=np.int64)
                pv = np.asarray(pv)
                for c, t, raw in r.messages(connections=[conns["/livox/point"]]):
                    hit = [w for w in remaining if abs(w - t) <= tol and w not in done]
                    if not hit:
                        continue
                    ts_ns = min(hit, key=lambda w: abs(w - t))
                    m = ts_store.deserialize_cdr(raw, c.msgtype)
                    xyz = pc2_to_xyz(m)
                    if xyz is None or len(xyz) < 2000:
                        continue
                    rec = dict(want[ts_ns])
                    tag = f"trial{rec['trial']:03d}_{rec['query']}"
                    npts = write_pcd(os.path.join(a.out, tag + ".pcd"), xyz)
                    j = int(np.argmin(np.abs(pt - t)))
                    rec.update(
                        {
                            "pcd": tag + ".pcd",
                            "n_points": npts,
                            "scan_dt_ms": (t - ts_ns) / 1e6,
                            "pose_dt_ms": (int(pt[j]) - t) / 1e6,
                            "gt_pose": [float(x) for x in pv[j]],
                            "bag": os.path.basename(bag),
                        }
                    )
                    done[ts_ns] = rec
        except Exception as exc:
            print(f"  skip {os.path.basename(bag)}: {exc}", file=sys.stderr)

    trials = sorted(done.values(), key=lambda r: r["trial"])
    miss = [w["query"] for t, w in want.items() if t not in done]
    with open(os.path.join(a.out, "trials.json"), "w", encoding="utf-8") as f:
        json.dump(trials, f, indent=1)
    print(f"[export] {len(trials)}/{len(want)} 帧成功,写到 {a.out}")
    if miss:
        print(f"  ⚠ 这 {len(miss)} 帧没配到雷达帧: {miss[:5]}")
    if trials:
        d = np.array([r["scan_dt_ms"] for r in trials])
        print(
            f"  scan_dt: |max| {np.abs(d).max():.1f} ms   点数中位 "
            f"{int(np.median([r['n_points'] for r in trials]))}"
        )


if __name__ == "__main__":
    main()
