import os
import sys
import time
import shutil
import argparse
import subprocess
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.append(root_dir)


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset",
        default=None,
        help="dataset selector (YAML stem or name); sets DATASET for config",
    )
    p.add_argument(
        "--target-root",
        default="datasets",
        help="output root, relative to repo (default: datasets)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve config and scan raw recordings, but copy nothing",
    )
    return p.parse_args()


ARGS = _parse_args()
if ARGS.dataset:
    os.environ["DATASET"] = ARGS.dataset

import numpy as np

from core.config import ROOT_DIR, TARGET_SCENE, CFG
from core.vpr_core import pose_manager, get_all_images_recursive

MAX_WORKERS = min(32, (os.cpu_count() or 4) * 2)

_B = CFG.dataset.build
MIN_DIST = float(_B.min_dist)
MIN_ANGLE = float(_B.min_angle)
INTERVALS = {
    "train": int(_B.intervals.train),
    "val": int(_B.intervals.val),
    "test": int(_B.intervals.test),
}
MAP_BOUNDARY_DIST = float(getattr(_B, "map_boundary_dist", 15.0))
IMAGE_SUBDIR = getattr(_B, "image_subdir", "undistorted_images")
FULL_FRAME_RECORDINGS = list(getattr(_B, "full_frame_recordings", []) or [])
RAW_ROOT = (
    _B.raw_root if os.path.isabs(_B.raw_root) else os.path.join(ROOT_DIR, _B.raw_root)
)


def _split_list(name):
    return list(getattr(_B.splits, name, []) or [])


SPLITS = {k: _split_list(k) for k in ("database", "train", "val", "test")}


def copy_meta_files(src_dir, dst_dir):

    meta_files = [
        "GNSSPoses.txt",
        "Transformations.txt",
        "pose.txt",
        "calib.yaml",
    ]
    copied = 0
    for meta_file in meta_files:
        meta_src = os.path.join(src_dir, meta_file)
        if os.path.exists(meta_src):
            shutil.copy2(meta_src, dst_dir)
            copied += 1
    if copied == 0:
        print(
            f"  ⚠️ no pose/calib sidecar found in {src_dir} -- if the build "
            f"later reports 0 candidate sequences, this is why"
        )


def _purge_split_outputs(parent, keep):

    if not keep:
        print(
            f"⚠️  Refusing to clear {os.path.relpath(parent, ROOT_DIR)}: "
            f"the split lists no recordings (missing or misspelt YAML key?)"
        )
        return
    if not os.path.isdir(parent):
        return
    for name in sorted(os.listdir(parent)):
        path = os.path.join(parent, name)
        if os.path.isdir(path):
            shutil.rmtree(path)
            if name not in keep:
                print(f"🧹 Removed stale recording: {os.path.relpath(path, ROOT_DIR)}")


def _thread_copy_task(src, dst_dir):

    dst = os.path.join(dst_dir, os.path.basename(src))
    if os.path.lexists(dst):
        os.remove(dst)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst_dir)


def _thread_filter_query(qi, q_img, q_pose, scene_db_poses, boundary_dist):
    for db_pose in scene_db_poses:
        dist = pose_manager.calculate_physical_distance(q_img, q_pose, db_pose)
        if dist is not None and dist <= boundary_dist:
            return q_img
    return None


def _run_ill_queries(raw_dir, out_dir, debug_dir, split):
    cmd = [
        sys.executable,
        "tools/generate_ill_queries.py",
        "--input",
        raw_dir,
        "--output",
        out_dir,
        "--debug",
        debug_dir,
        "--workers",
        str(MAX_WORKERS),
    ]
    if FULL_FRAME_RECORDINGS:
        cmd += ["--full-frame-recordings", ",".join(FULL_FRAME_RECORDINGS)]
    subprocess.run(
        cmd, cwd=ROOT_DIR, check=True, env={**os.environ, "QUERY_SPLIT": split}
    )


def _dry_run_report(recording_dirs):
    print(
        f"🔎 [DRY-RUN] Dataset '{TARGET_SCENE}' (pose_format={CFG.dataset.pose_format})"
    )
    print(f"   raw_root      : {RAW_ROOT}")
    print(f"   image_subdir  : {IMAGE_SUBDIR}")
    print(f"   min_dist/angle: {MIN_DIST} m / {MIN_ANGLE} deg")
    print(f"   intervals     : {INTERVALS}")
    print(f"   map_boundary  : {MAP_BOUNDARY_DIST} m")
    print(f"   splits        : {SPLITS}")
    print(f"   recordings    : {recording_dirs}")
    for rec in recording_dirs:
        roles = [r for r in ("database", "train", "val", "test") if rec in SPLITS[r]]
        img_dir = os.path.join(RAW_ROOT, rec, IMAGE_SUBDIR)
        n_imgs = len(get_all_images_recursive(img_dir)) if os.path.isdir(img_dir) else 0
        status = (
            "✅" if (roles and n_imgs) else ("⚠️ no images" if roles else "— unused")
        )
        roles_str = ",".join(roles) if roles else "-"
        print(f"     - {rec:24s} roles={roles_str:<20} images={n_imgs:<6} {status}")
    print("🔎 [DRY-RUN] No files copied.")


def build_paper_dataset(
    target_rel_dir="datasets", min_dist=MIN_DIST, min_angle=MIN_ANGLE, dry_run=False
):
    scene_name = TARGET_SCENE
    if not os.path.isdir(RAW_ROOT):
        raise FileNotFoundError(f"raw_root does not exist: {RAW_ROOT}")

    recording_dirs = sorted(
        d for d in os.listdir(RAW_ROOT) if os.path.isdir(os.path.join(RAW_ROOT, d))
    )

    if dry_run:
        _dry_run_report(recording_dirs)
        return

    print(
        f"🚀 Building SPARE dataset [{scene_name}] (map boundary={MAP_BOUNDARY_DIST} m)..."
    )
    print(f"⚡ Multithreaded (Max Workers: {MAX_WORKERS})")

    scene_target_dir = os.path.join(ROOT_DIR, target_rel_dir, scene_name)

    db_out = os.path.join(scene_target_dir, "database")
    queries_raw_train = os.path.join(scene_target_dir, "queries_train_raw")
    queries_raw_val = os.path.join(scene_target_dir, "queries_val_raw")
    queries_raw_test = os.path.join(scene_target_dir, "queries_test_raw")
    queries_train = os.path.join(scene_target_dir, "queries_train")
    queries_val = os.path.join(scene_target_dir, "queries_val")
    queries_test = os.path.join(scene_target_dir, "queries_test")

    for d in [
        db_out,
        queries_raw_train,
        queries_raw_val,
        queries_raw_test,
        queries_train,
        queries_val,
        queries_test,
    ]:
        os.makedirs(d, exist_ok=True)

    for split, dirs in (
        ("database", [db_out]),
        (
            "train",
            [
                queries_raw_train,
                queries_train,
                os.path.join(scene_target_dir, "queries_debug_train"),
            ],
        ),
        (
            "val",
            [
                queries_raw_val,
                queries_val,
                os.path.join(scene_target_dir, "queries_debug_val"),
            ],
        ),
        (
            "test",
            [
                queries_raw_test,
                queries_test,
                os.path.join(scene_target_dir, "queries_debug_test"),
            ],
        ),
    ):
        for parent in dirs:
            _purge_split_outputs(parent, SPLITS[split])

    scene_db_poses = []
    scene_cache = []
    total_db_frames = 0
    total_train_queries = 0
    total_val_queries = 0
    total_test_queries = 0

    print(f"\n🔍 [DEBUG] Scene: {scene_name}, found recordings: {recording_dirs}")

    for recording_name in recording_dirs:
        is_db = recording_name in SPLITS["database"]
        is_train = recording_name in SPLITS["train"]
        is_val = recording_name in SPLITS["val"]
        is_test = recording_name in SPLITS["test"]

        if not any([is_db, is_train, is_val, is_test]):
            print(
                f"🔍 [DEBUG] ⚠️ Skipping: '{recording_name}' is not listed in any split."
            )
            continue

        recording_bakup_path = os.path.join(RAW_ROOT, recording_name)
        rel_record_path = recording_name
        img_dir = os.path.join(recording_bakup_path, IMAGE_SUBDIR)

        print(
            f"🔍 [DEBUG] Matched target. Checking image dir: {img_dir} | exists: {os.path.exists(img_dir)}"
        )

        all_images = get_all_images_recursive(img_dir)
        if not all_images:
            print(f"🔍 [DEBUG] ❌ Blocked: found 0 images under {img_dir}; skipping.")
            continue
        else:
            print(f"🔍 [DEBUG] ✅ Found {len(all_images)} images under this directory.")

        roles = []
        if is_db:
            roles.append("🗺️ DB")
        if is_train:
            roles.append("🎓 Train")
        if is_val:
            roles.append("⚖️ Val")
        if is_test:
            roles.append("🕵️ Test")
        print(f"\n🌍 Scanning: {rel_record_path} [{' + '.join(roles)}]")

        downsampled_images = []
        downsampled_poses = []
        last_global_pose = None

        for img in tqdm(
            all_images, desc="  📏 Translation/rotation downsampling", leave=False
        ):
            info = pose_manager.get_pose_info(img)
            if not info or not info.get("global"):
                continue
            curr_global_pose = info["global"]

            if last_global_pose is None:
                downsampled_images.append(img)
                downsampled_poses.append(curr_global_pose)
                last_global_pose = curr_global_pose
            else:
                dist = pose_manager.calculate_physical_distance(
                    img, last_global_pose, curr_global_pose
                )
                angle = pose_manager.calculate_angular_distance(
                    last_global_pose, curr_global_pose
                )
                is_dist_ok = dist is not None and dist >= min_dist
                is_angle_ok = angle is not None and angle >= min_angle
                if is_dist_ok or is_angle_ok:
                    downsampled_images.append(img)
                    downsampled_poses.append(curr_global_pose)
                    last_global_pose = curr_global_pose

        if not downsampled_images:
            continue

        role_query_indices = {}
        if is_train:
            role_query_indices["train"] = list(
                range(0, len(downsampled_images), INTERVALS["train"])
            )
        if is_val:
            if is_train:
                taken = set(role_query_indices["train"])
                pool = [i for i in range(len(downsampled_images)) if i not in taken]
                stride = max(
                    1, (len(pool) * INTERVALS["val"]) // max(1, len(downsampled_images))
                )
                role_query_indices["val"] = pool[::stride]
            else:
                role_query_indices["val"] = list(
                    range(0, len(downsampled_images), INTERVALS["val"])
                )
        if is_test:
            role_query_indices["test"] = list(
                range(0, len(downsampled_images), INTERVALS["test"])
            )

        db_indices = []
        if is_db:
            db_indices = list(range(len(downsampled_images)))
            for i in db_indices:
                scene_db_poses.append(downsampled_poses[i])

        scene_cache.append(
            {
                "rel_record_path": rel_record_path,
                "recording_bakup_path": recording_bakup_path,
                "is_db": is_db,
                "is_train": is_train,
                "is_val": is_val,
                "is_test": is_test,
                "images": downsampled_images,
                "poses": downsampled_poses,
                "db_indices": db_indices,
                "role_query_indices": role_query_indices,
            }
        )

    heldout_basenames = set()
    for data in scene_cache:
        for _hrole in ("val", "test"):
            for qi in data["role_query_indices"].get(_hrole, []):
                heldout_basenames.add(os.path.basename(data["images"][qi]))

    for data in scene_cache:
        if data["is_db"] and data["db_indices"]:
            scene_db_out = os.path.join(db_out, data["rel_record_path"])
            scene_db_img_out = os.path.join(scene_db_out, IMAGE_SUBDIR)
            os.makedirs(scene_db_img_out, exist_ok=True)

            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = [
                    executor.submit(
                        _thread_copy_task, data["images"][i], scene_db_img_out
                    )
                    for i in data["db_indices"]
                ]
                for _ in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"  💾 Saving DB concurrently: {data['rel_record_path']}",
                    leave=False,
                ):
                    pass

            copy_meta_files(data["recording_bakup_path"], scene_db_out)
            total_db_frames += len(data["db_indices"])

        for _role, _role_idx in sorted(data["role_query_indices"].items()):
            if not _role_idx:
                continue
            valid_queries = []

            if not scene_db_poses:
                print(
                    f"  ❌ Warning: {data['rel_record_path']} has no DB poses to reference; dropping all queries."
                )
                continue

            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = [
                    executor.submit(
                        _thread_filter_query,
                        qi,
                        data["images"][qi],
                        data["poses"][qi],
                        scene_db_poses,
                        MAP_BOUNDARY_DIST,
                    )
                    for qi in _role_idx
                ]
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"  🔍 Concurrent map-boundary filtering: {data['rel_record_path']}",
                    leave=False,
                ):
                    res = future.result()
                    if res:
                        valid_queries.append(res)

            print(
                f"  ✂️ Filter result[{_role}]: sampled {len(_role_idx)} -> remaining after boundary filter {len(valid_queries)}"
            )

            if _role == "train" and heldout_basenames:
                _before = len(valid_queries)
                valid_queries = [
                    q
                    for q in valid_queries
                    if os.path.basename(q) not in heldout_basenames
                ]
                _dropped = _before - len(valid_queries)
                if _dropped:
                    print(
                        f"  🔒 Held-out isolation: dropped {_dropped} train frames "
                        f"that are also val/test queries (no train/test leakage)."
                    )

            if not valid_queries:
                continue

            def _concurrent_copy_valid_queries(queries_list, target_raw_base):
                scene_raw_out = os.path.join(target_raw_base, data["rel_record_path"])
                scene_raw_img_out = os.path.join(scene_raw_out, IMAGE_SUBDIR)
                os.makedirs(scene_raw_img_out, exist_ok=True)

                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    futures = [
                        executor.submit(_thread_copy_task, q, scene_raw_img_out)
                        for q in queries_list
                    ]
                    for _ in as_completed(futures):
                        pass
                copy_meta_files(data["recording_bakup_path"], scene_raw_out)

            if _role == "train":
                total_train_queries += len(valid_queries)
                _concurrent_copy_valid_queries(valid_queries, queries_raw_train)
                print(f"  ✅ Loaded {len(valid_queries)} frames into Train.")
            elif _role == "val":
                total_val_queries += len(valid_queries)
                _concurrent_copy_valid_queries(valid_queries, queries_raw_val)
                print(f"  ✅ Loaded {len(valid_queries)} frames into Val.")
            elif _role == "test":
                total_test_queries += len(valid_queries)
                _concurrent_copy_valid_queries(valid_queries, queries_raw_test)
                print(f"  ✅ Loaded {len(valid_queries)} frames into Test.")

    print(f"\n🗺️ Scene [{scene_name}] sandbox complete. Summary:")
    print(f"   - 🗺️ Database frames: {total_db_frames}")
    print(
        f"   - 🎓 Train queries: {total_train_queries} (interval {INTERVALS['train']})"
    )
    print(f"   - ⚖️ Val queries: {total_val_queries} (interval {INTERVALS['val']})")
    print(f"   - 🕵️ Test queries: {total_test_queries} (interval {INTERVALS['test']})")

    empty_roles = [
        name
        for name, count in (
            ("database", total_db_frames),
            ("train", total_train_queries),
            ("val", total_val_queries),
            ("test", total_test_queries),
        )
        if count == 0
    ]
    if empty_roles:
        raise RuntimeError(
            "dataset build produced zero frames for: "
            + ", ".join(empty_roles)
            + ". Check recording names, images and pose sidecars in raw_root."
        )

    if total_train_queries > 0:
        print("\n📸 Generating Train close-up queries...")
        _run_ill_queries(
            queries_raw_train,
            queries_train,
            os.path.join(scene_target_dir, "queries_debug_train"),
            "train",
        )
    if total_val_queries > 0:
        _run_ill_queries(
            queries_raw_val,
            queries_val,
            os.path.join(scene_target_dir, "queries_debug_val"),
            "val",
        )
    if total_test_queries > 0:
        _run_ill_queries(
            queries_raw_test,
            queries_test,
            os.path.join(scene_target_dir, "queries_debug_test"),
            "test",
        )

    print(
        "\n✅ Dataset build finished (with translation/rotation continuity protection)."
    )


if __name__ == "__main__":
    if not ARGS.dry_run:
        print("\n🧹 (Old-data cleanup is left to tools/clean_data.py if needed.)")
        time.sleep(0.5)
    build_paper_dataset(target_rel_dir=ARGS.target_root, dry_run=ARGS.dry_run)
