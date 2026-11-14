import concurrent.futures
import glob
import os
import sys
import threading

import cv2
import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import VPR_MODEL_ID, TARGET_SCENE, SEQUENCE_ASSET_FORMAT
from core.vpr_core import VPRService, get_all_images_recursive


def async_save_assets(payload, save_path):

    tmp_path = f"{save_path}.tmp.{os.getpid()}.{threading.get_ident()}"
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, save_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def build_offline_sequences(dataset_path=f"datasets/{TARGET_SCENE}"):

    print(f"--- 🚀 Building offline sequence assets for {VPR_MODEL_ID} ---")

    db_dir = os.path.join(dataset_path, "database")
    if not os.path.exists(db_dir):
        raise FileNotFoundError(f"Database directory not found: {db_dir}")
    all_images = get_all_images_recursive(db_dir)
    rel_paths = [os.path.relpath(path, dataset_path) for path in all_images]

    output_dir = os.path.join("assets", "sequence_assets", TARGET_SCENE)
    os.makedirs(output_dir, exist_ok=True)
    interrupted = glob.glob(
        os.path.join(output_dir, "**", "*.pt.tmp.*"), recursive=True
    )
    for path in interrupted:
        os.remove(path)
    if interrupted:
        print(f"[sequence assets] removed {len(interrupted)} interrupted temp files")
    requested_force = os.getenv("FORCE_REBUILD_SEQUENCE_ASSETS", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    force_rebuild = requested_force
    save_paths = [os.path.join(output_dir, rel_path + ".pt") for rel_path in rel_paths]
    missing_count = sum(not os.path.exists(path) for path in save_paths)
    if not force_rebuild and missing_count == 0:
        print(f"✅ Done. 0 built, {len(rel_paths)} presence-reused.")
        return

    service = VPRService()
    cropper = service.cropper
    print("--- 📦 Ensuring global database anchor features exist ---")
    service._get_or_compute_db_features(dataset_path)

    io_pool = concurrent.futures.ThreadPoolExecutor(max_workers=8)
    asset_futures = []
    rebuilt = 0
    for img_path, rel_path in tqdm(
        list(zip(all_images, rel_paths)), desc="Offline Sequence Compiling"
    ):
        save_path = os.path.join(output_dir, rel_path + ".pt")
        if os.path.exists(save_path) and not force_rebuild:
            continue

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        img_cv = cv2.imread(img_path)
        if img_cv is None:
            raise RuntimeError(f"unreadable database image: {img_path}")
        feats = cropper.extract_image(img_cv, image_path=img_path)
        payload = {
            "format": SEQUENCE_ASSET_FORMAT,
            "source_rel_path": rel_path.replace(os.sep, "/"),
            "lightglue_feats": [cropper.pack_offline_features(feats)],
            "rel_seq_paths": [rel_path.replace(os.sep, "/")],
        }
        del feats
        asset_futures.append(io_pool.submit(async_save_assets, payload, save_path))
        rebuilt += 1

    io_pool.shutdown(wait=True)
    for future in asset_futures:
        future.result()

    missing = [
        rel
        for rel in rel_paths
        if not os.path.isfile(os.path.join(output_dir, rel + ".pt"))
    ]
    if missing:
        raise RuntimeError(
            f"sequence asset build is incomplete: {len(missing)} missing, first={missing[0]}"
        )
    print(f"\n✅ Done. {rebuilt} built, {len(rel_paths) - rebuilt} presence-reused.")


if __name__ == "__main__":
    build_offline_sequences()
