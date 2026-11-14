#!/usr/bin/env python


import argparse
import csv
import os
import shutil

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MSLS_ROOT = os.path.join(ROOT, "datasets", "public_benchmarks", "msls")
SPLIT_DIR = os.path.join(
    ROOT, "3rdparty", "Pair-VPR", "pairvpr", "datasets", "datasetfiles", "msls_val"
)
OUT_ROOT = os.path.join(ROOT, "datasets", "database_raw_processed", "MSLS")

ID_BASE = {
    "recording_mapping": 10_000_000,
    "recording_val": 20_000_000,
    "recording_test": 30_000_000,
}
CITY_EAST_OFFSET = {"cph": 0.0, "sf": 1_000_000.0}
VAL_EVERY_NTH = 5
VAL_PHASE = 2


def load_coords():
    coords = {}
    for city in ("cph", "sf"):
        for part in ("database", "query"):
            path = os.path.join(MSLS_ROOT, "train_val", city, part, "postprocessed.csv")
            with open(path, newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    coords[(city, part, row["key"])] = (
                        float(row["easting"]) + CITY_EAST_OFFSET[city],
                        float(row["northing"]),
                    )
    return coords


def parse_rel(rel):
    parts = rel.split("/")
    city, part, key = parts[1], parts[2], os.path.splitext(parts[-1])[0]
    return city, part, key


def write_recording(name, items, force):
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
        src = os.path.join(MSLS_ROOT, rel)
        if not os.path.isfile(src):
            raise FileNotFoundError(src)
        out = os.path.join(dst, f"{image_id}.jpg")
        if not os.path.isfile(out):
            shutil.copy2(src, out)
        lines.append(f"{image_id} {east:.6f} {north:.6f} 0.000000 0 0 0 1")
    tmp = pose_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    os.replace(tmp, pose_path)
    print(f"[done] {name}: {len(items)} images -> {dst}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset", default="msls", help="accepted for reproduce.py's prepare hook"
    )
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.dataset.lower() != "msls":
        raise SystemExit(f"this tool stages MSLS only, got --dataset {a.dataset}")

    db = [
        str(x)
        for x in np.load(
            os.path.join(SPLIT_DIR, "msls_val_dbImages.npy"), allow_pickle=True
        )
    ]
    q_all = [
        str(x)
        for x in np.load(
            os.path.join(SPLIT_DIR, "msls_val_qImages.npy"), allow_pickle=True
        )
    ]
    q_idx = np.load(os.path.join(SPLIT_DIR, "msls_val_qIdx.npy"), allow_pickle=True)
    queries = [q_all[int(i)] for i in q_idx]
    coords = load_coords()

    def items_for(rels):
        out = []
        for k, rel in enumerate(rels):
            east, north = coords[parse_rel(rel)]
            out.append((k, rel, east, north))
        return out

    mapping = items_for(db)
    val = [item for item in mapping if item[0] % VAL_EVERY_NTH == VAL_PHASE]
    test = items_for(queries)
    print(f"[msls] db {len(mapping)}, val {len(val)}, test queries {len(test)}")
    write_recording("recording_mapping", mapping, a.force)
    write_recording("recording_val", val, a.force)
    write_recording("recording_test", test, a.force)


if __name__ == "__main__":
    main()
