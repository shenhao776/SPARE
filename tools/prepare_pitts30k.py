#!/usr/bin/env python


import argparse
import math
import os
import re
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCHIVE_ROOT = os.path.expanduser("~/datasets/Pittsburgh250k")
STAGED_ROOT = os.path.join(ROOT, "datasets", "public_benchmarks", "pitts30k")
SPLIT_MAT = os.path.join(
    ARCHIVE_ROOT, "netvlad_splits", "datasets", "pitts30k_test.mat"
)
OUT_ROOT = os.path.join(ROOT, "datasets", "database_raw_processed", "Pitts30k")

ID_BASE = {
    "recording_mapping": 10_000_000,
    "recording_val": 20_000_000,
    "recording_test": 30_000_000,
}
VAL_EVERY_NTH_LOCATION = 5
VAL_PHASE = 2

_VIEW = re.compile(r"pitch(\d+)_yaw(\d+)")


def quat_from_pitch_yaw(pitch_deg, yaw_deg):

    cy, sy = math.cos(math.radians(yaw_deg) / 2), math.sin(math.radians(yaw_deg) / 2)
    cp, sp = math.cos(math.radians(pitch_deg) / 2), math.sin(
        math.radians(pitch_deg) / 2
    )
    qw = cy * cp
    qx = cy * sp
    qy = sy * sp
    qz = sy * cp
    return qx, qy, qz, qw


def view_angles(name):
    m = _VIEW.search(name)
    if not m:
        return 0.0, 0.0
    pitch_idx, yaw_idx = int(m.group(1)), int(m.group(2))
    return (30.0 if pitch_idx >= 2 else 0.0), (yaw_idx - 1) * 30.0


def resolve_image(rel, sub=""):

    for base in (STAGED_ROOT, ARCHIVE_ROOT):
        p = os.path.join(base, sub, rel) if sub else os.path.join(base, rel)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"{sub}/{rel}: not under {STAGED_ROOT} nor {ARCHIVE_ROOT}")


def load_split():
    import numpy as np
    from scipy.io import loadmat

    db = loadmat(SPLIT_MAT, squeeze_me=True, struct_as_record=False)["dbStruct"]
    utm_db = np.asarray(db.utmDb, dtype=float)
    utm_q = np.asarray(db.utmQ, dtype=float)
    if utm_db.shape[0] == 2:
        utm_db = utm_db.T
    if utm_q.shape[0] == 2:
        utm_q = utm_q.T
    return (
        [str(x) for x in db.dbImageFns],
        utm_db,
        [str(x) for x in db.qImageFns],
        utm_q,
    )


def write_recording(name, items, force, sub=""):

    dst = os.path.join(OUT_ROOT, name)
    pose_path = os.path.join(dst, "pose.txt")
    if os.path.isdir(dst) and not force:
        have = sum(1 for f in os.listdir(dst) if f.endswith(".jpg"))
        if have == len(items) and os.path.isfile(pose_path):
            print(f"[skip] {name}: {have} images + pose.txt already present")
            return
    if force and os.path.isdir(dst):
        shutil.rmtree(dst)
    os.makedirs(dst, exist_ok=True)
    base = ID_BASE[name]
    lines = []
    for k, rel, east, north in items:
        image_id = base + k
        src = resolve_image(rel, sub)
        out = os.path.join(dst, f"{image_id}.jpg")
        if not os.path.isfile(out):
            shutil.copy2(src, out)
        pitch, yaw = view_angles(rel)
        qx, qy, qz, qw = quat_from_pitch_yaw(pitch, yaw)
        lines.append(
            f"{image_id} {east:.6f} {north:.6f} 0.000000 "
            f"{qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}"
        )
    tmp = pose_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    os.replace(tmp, pose_path)
    print(f"[done] {name}: {len(items)} images -> {dst}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset", default="pitts30k", help="accepted for reproduce.py's prepare hook"
    )
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.dataset.lower() != "pitts30k":
        raise SystemExit(f"this tool stages Pitts30k only, got --dataset {a.dataset}")
    if not os.path.isfile(SPLIT_MAT):
        raise SystemExit(f"missing NetVLAD split: {SPLIT_MAT}")

    db_names, utm_db, q_names, utm_q = load_split()
    mapping = [
        (k, rel, float(e), float(n))
        for k, (rel, (e, n)) in enumerate(zip(db_names, utm_db))
    ]
    loc_index = {}
    for _, _, e, n in mapping:
        key = (round(e, 2), round(n, 2))
        loc_index.setdefault(key, len(loc_index))
    val = [
        item
        for item in mapping
        if loc_index[(round(item[2], 2), round(item[3], 2))] % VAL_EVERY_NTH_LOCATION
        == VAL_PHASE
    ]
    test = [
        (k, rel, float(e), float(n))
        for k, (rel, (e, n)) in enumerate(zip(q_names, utm_q))
    ]
    print(
        f"[pitts30k] db {len(mapping)} views / {len(loc_index)} locations, "
        f"val {len(val)} views, test queries {len(test)}"
    )
    write_recording("recording_mapping", mapping, a.force)
    write_recording("recording_val", val, a.force)
    write_recording("recording_test", test, a.force, sub="queries_real")


if __name__ == "__main__":
    main()
