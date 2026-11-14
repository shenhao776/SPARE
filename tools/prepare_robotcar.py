#!/usr/bin/env python3


import argparse
import bisect
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROLES = {
    "mapping": "recording_mapping",
    "val": "recording_1",
    "test": "recording_2",
    "night": "recording_night",
}
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
MANIFEST_NAME = ".robotcar_prepare_manifest.json"
FORMAT_VERSION = 1


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="robotcar")
    parser.add_argument("--source-root")
    parser.add_argument("--output-root")
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    return parser.parse_args()


def _load_config(dataset):
    sys.path.insert(0, ROOT)
    from core.config import resolve_dataset

    config = resolve_dataset(dataset)
    prepare = dict(config.get("prepare") or {})
    build = dict(config.get("build") or {})
    if not prepare:
        raise SystemExit(f"[{dataset}] missing prepare: block in dataset YAML")
    source_root = prepare.get("source_root")
    output_root = build.get("raw_root")
    if output_root and not os.path.isabs(output_root):
        output_root = os.path.join(ROOT, output_root)
    return config, prepare, source_root, output_root


def _session_images(source_root, session):
    image_dir = os.path.join(source_root, session, "stereo", "centre")
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"missing front-camera directory: {image_dir}")
    images = []
    for name in os.listdir(image_dir):
        if not name.lower().endswith(IMAGE_EXTENSIONS):
            continue
        try:
            timestamp_us = int(os.path.splitext(name)[0])
        except ValueError:
            continue
        images.append((timestamp_us, os.path.join(image_dir, name)))
    images.sort()
    if not images:
        raise RuntimeError(f"no timestamped images in {image_dir}")
    return images


def _rtk_archive(source_root, prepare):
    configured = prepare.get("rtk_archive")
    path = configured or os.path.join(source_root, "rtk.zip")
    if not os.path.isabs(path):
        path = os.path.join(source_root, path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"missing RobotCar RTK archive: {path}")
    return path


def _load_rtk(source_root, session, prepare):
    archive = _rtk_archive(source_root, prepare)
    member = f"{session}/rtk.csv"
    with zipfile.ZipFile(archive) as handle:
        try:
            raw = handle.open(member)
        except KeyError as exc:
            raise FileNotFoundError(f"{member} is missing from {archive}") from exc
        rows = []
        with raw:
            text = (line.decode("utf-8") for line in raw)
            for row in csv.DictReader(text):
                try:
                    values = (
                        int(row["timestamp"]),
                        float(row["easting"]),
                        float(row["northing"]),
                        -float(row["down"]),
                        float(row["roll"]),
                        float(row["pitch"]),
                        float(row["yaw"]),
                    )
                except (KeyError, TypeError, ValueError):
                    continue
                if all(math.isfinite(value) for value in values[1:]):
                    rows.append(values)
    rows.sort()
    if not rows:
        raise RuntimeError(f"no valid RTK rows for {session}")
    return rows


def _angle_difference(a, b):
    return abs((a - b + math.pi) % (2.0 * math.pi) - math.pi)


def _match_and_sample(images, rtk_rows, spacing_m, angle_deg, max_delta_ms):
    timestamps = [row[0] for row in rtk_rows]
    max_delta_us = int(max_delta_ms * 1000.0)
    angle_rad = math.radians(angle_deg)
    selected = []
    rejected = 0
    last_pose = None
    for image_timestamp, image_path in images:
        index = bisect.bisect_left(timestamps, image_timestamp)
        candidates = [
            candidate
            for candidate in (index - 1, index)
            if 0 <= candidate < len(rtk_rows)
        ]
        if not candidates:
            rejected += 1
            continue
        nearest = min(
            candidates,
            key=lambda candidate: abs(timestamps[candidate] - image_timestamp),
        )
        pose = rtk_rows[nearest]
        if abs(pose[0] - image_timestamp) > max_delta_us:
            rejected += 1
            continue
        if last_pose is not None:
            distance = math.hypot(pose[1] - last_pose[1], pose[2] - last_pose[2])
            rotation = _angle_difference(pose[6], last_pose[6])
            if distance < spacing_m and rotation < angle_rad:
                continue
        selected.append((image_timestamp, image_path, pose))
        last_pose = pose
    return selected, rejected


def _quaternion(roll, pitch, yaw):
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def _bayer_code(pattern):
    codes = {
        "bggr": cv2.COLOR_BayerBG2BGR,
        "gbrg": cv2.COLOR_BayerGB2BGR,
        "grbg": cv2.COLOR_BayerGR2BGR,
        "rggb": cv2.COLOR_BayerRG2BGR,
    }
    try:
        return codes[pattern.lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported Bayer pattern: {pattern}") from exc


def _convert_image(task):
    source, target, bayer_pattern, jpeg_quality = task
    image = cv2.imread(source, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"OpenCV could not read {source}")
    if image.ndim == 2:
        image = cv2.cvtColor(image, _bayer_code(bayer_pattern))
    elif image.ndim != 3 or image.shape[2] not in (3, 4):
        raise RuntimeError(f"unsupported image shape {image.shape}: {source}")
    if image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    temporary = target + f".tmp.{os.getpid()}.jpg"
    if not cv2.imwrite(temporary, image, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]):
        raise RuntimeError(f"failed to write {target}")
    os.replace(temporary, target)


def _atomic_write_json(path, payload):
    temporary = f"{path}.tmp.{os.getpid()}"
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def _complete_output(output_root, sessions, manifest):
    if not manifest:
        return False
    counts = manifest.get("counts") or {}
    for role, output_name in ROLES.items():
        expected = counts.get(role, {}).get("selected")
        image_dir = os.path.join(output_root, output_name, "images")
        pose_file = os.path.join(output_root, output_name, "pose.txt")
        if (
            expected is None
            or not os.path.isfile(pose_file)
            or not os.path.isdir(image_dir)
        ):
            return False
        actual = sum(name.lower().endswith(".jpg") for name in os.listdir(image_dir))
        if actual != expected:
            return False
    return True


def _load_manifest(output_root):
    path = os.path.join(output_root, MANIFEST_NAME)
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def _atomic_install(staging_root, output_root):
    backup = output_root + f".previous.{os.getpid()}"
    if os.path.exists(backup):
        shutil.rmtree(backup)
    if os.path.exists(output_root):
        os.replace(output_root, backup)
    try:
        os.replace(staging_root, output_root)
    except Exception:
        if os.path.exists(backup) and not os.path.exists(output_root):
            os.replace(backup, output_root)
        raise
    if os.path.exists(backup):
        shutil.rmtree(backup)


def _grid_cells(selected, cell_m=5.0):
    return {
        (round(item[2][1] / cell_m), round(item[2][2] / cell_m)) for item in selected
    }


def analyze(source_root, prepare):
    sessions = sorted(
        name
        for name in os.listdir(source_root)
        if name[:4].isdigit() and os.path.isdir(os.path.join(source_root, name))
    )
    spacing_m = float(prepare.get("spacing_m", 1.0))
    angle_deg = float(prepare.get("angle_deg", 10.0))
    max_delta_ms = float(prepare.get("max_pose_delta_ms", 100.0))
    grids = {}
    print("session,images,matched_sampled,grid_5m")
    for session in sessions:
        images = _session_images(source_root, session)
        selected, _ = _match_and_sample(
            images,
            _load_rtk(source_root, session, prepare),
            spacing_m,
            angle_deg,
            max_delta_ms,
        )
        grids[session] = _grid_cells(selected)
        print(f"{session},{len(images)},{len(selected)},{len(grids[session])}")
    print("\npairwise 5 m grid overlap: rows are query coverage by columns")
    print("query\\map," + ",".join(sessions))
    for query in sessions:
        values = []
        for mapping in sessions:
            denominator = max(1, len(grids[query]))
            values.append(f"{len(grids[query] & grids[mapping]) / denominator:.3f}")
        print(query + "," + ",".join(values))


def _build_role(
    staging_root, source_root, role, output_name, session, parameters, prepare, workers
):

    images = _session_images(source_root, session)
    selected, rejected = _match_and_sample(
        images,
        _load_rtk(source_root, session, prepare),
        parameters["spacing_m"],
        parameters["angle_deg"],
        parameters["max_pose_delta_ms"],
    )
    recording_root = os.path.join(staging_root, output_name)
    image_root = os.path.join(recording_root, "images")
    os.makedirs(image_root)
    tasks = []
    pose_lines = ["timestamp tx ty tz qx qy qz qw\n"]
    for timestamp_us, image_path, pose in selected:
        qx, qy, qz, qw = _quaternion(pose[4], pose[5], pose[6])
        pose_lines.append(
            f"{timestamp_us * 1000} {pose[1]:.6f} {pose[2]:.6f} {pose[3]:.6f} "
            f"{qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}\n"
        )
        target = os.path.join(image_root, f"{timestamp_us}.jpg")
        tasks.append(
            (
                image_path,
                target,
                parameters["bayer_pattern"],
                parameters["jpeg_quality"],
            )
        )
    if not tasks:
        raise RuntimeError(f"{session}: no image has a usable RTK pose")
    with open(
        os.path.join(recording_root, "pose.txt"), "w", encoding="utf-8"
    ) as handle:
        handle.writelines(pose_lines)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        for index, _ in enumerate(executor.map(_convert_image, tasks), 1):
            if index % 1000 == 0 or index == len(tasks):
                print(f"[RobotCar prepare] {role}: {index}/{len(tasks)} RGB frames")
    return {
        "source_session": session,
        "source_images": len(images),
        "selected": len(selected),
        "pose_rejected": rejected,
    }


def prepare_dataset(source_root, output_root, prepare, force=False, workers=8):
    sessions = dict(prepare.get("sessions") or {})
    missing_roles = sorted(set(ROLES) - set(sessions))
    if missing_roles:
        raise SystemExit(f"prepare.sessions is missing: {', '.join(missing_roles)}")
    previous = _load_manifest(output_root)
    if not source_root or not os.path.isdir(source_root):
        if _complete_output(output_root, sessions, previous):
            print(
                f"[RobotCar prepare] source unavailable; using portable prepared input: {output_root}"
            )
            return
        raise FileNotFoundError(
            "RobotCar source root is missing. Pass --source-root or set "
            "ROBOTCAR_SOURCE_ROOT; the portable prepared input is also incomplete: "
            f"{output_root}"
        )

    parameters = {
        "sessions": sessions,
        "spacing_m": float(prepare.get("spacing_m", 1.0)),
        "angle_deg": float(prepare.get("angle_deg", 10.0)),
        "max_pose_delta_ms": float(prepare.get("max_pose_delta_ms", 100.0)),
        "bayer_pattern": str(prepare.get("bayer_pattern", "gbrg")),
        "jpeg_quality": int(prepare.get("jpeg_quality", 95)),
    }
    if not force and _complete_output(output_root, sessions, previous):
        print(f"[RobotCar prepare] already complete: {output_root}")
        return

    parent = os.path.dirname(output_root)
    os.makedirs(parent, exist_ok=True)
    staging_root = tempfile.mkdtemp(prefix="RobotCar.staging.", dir=parent)
    counts = {}
    try:
        for role, output_name in ROLES.items():
            counts[role] = _build_role(
                staging_root,
                source_root,
                role,
                output_name,
                sessions[role],
                parameters,
                prepare,
                workers,
            )
        manifest = {
            "format_version": FORMAT_VERSION,
            "source_root": source_root,
            "parameters": parameters,
            "counts": counts,
        }
        _atomic_write_json(os.path.join(staging_root, MANIFEST_NAME), manifest)
        _atomic_install(staging_root, output_root)
        staging_root = None
    finally:
        if staging_root and os.path.isdir(staging_root):
            shutil.rmtree(staging_root)
    print(f"[RobotCar prepare] portable input ready: {output_root}")


def main():
    args = _parse_args()
    _, prepare, configured_source, configured_output = _load_config(args.dataset)
    source_value = (
        args.source_root or os.environ.get("ROBOTCAR_SOURCE_ROOT") or configured_source
    )
    source_root = None
    if source_value:
        source_root = (
            source_value
            if os.path.isabs(source_value)
            else os.path.join(ROOT, source_value)
        )
        source_root = os.path.abspath(source_root)
    output_root = os.path.abspath(args.output_root or configured_output)
    if args.analyze:
        if not source_root:
            raise SystemExit("--analyze requires --source-root or ROBOTCAR_SOURCE_ROOT")
        analyze(source_root, prepare)
        return
    prepare_dataset(source_root, output_root, prepare, args.force, args.workers)


if __name__ == "__main__":
    main()
