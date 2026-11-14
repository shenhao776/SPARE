import os
import sys
import json
import faiss
import cv2
import random
from tqdm import tqdm
from PIL import Image
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import (
    data_tag,
    SEQUENCE_ASSET_FORMAT,
    SUCCESS_THRESHOLD_TRAIN,
    MIN_CANDIDATE_DIST,
    FEATURE_MATCH_MIN_RAW,
    GEOMETRY_KEYS,
    FEATURE_MATCH_MIN_INLIERS,
    FEATURE_HULL_MIN_INLIERS,
    RANSAC_REPROJ_THRESHOLD,
    STEP_FEATURE_TAU,
    STEP_MAX_DIST,
    STEP_BUDGET,
    SPATIAL_STEPS,
    TARGET_SCENE,
    TOP_K,
    TARGET_SCENE,
    VPR_MODEL_ID,
    DATA_ABLATION_RANDOM_WINDOW,
)

import core.vpr_core as vpr_core
from sl_agent.cache_fingerprint import (
    CHECK_ONLY,
    CHECK_REBUILD_EXIT,
    build_fingerprint,
    read_fingerprint,
    resume_reason,
    state_build_settings,
    validate_sequence_asset,
    write_fingerprint,
)
from sl_agent.offline_match_cache import OfflineQueryMatchCache
from core.vpr_core import (
    VPRService,
    pose_manager,
    get_database_spatial_sequence,
    get_all_images_recursive,
    get_base_transform,
)

QUERY_DATASET_PATH = f"datasets/{TARGET_SCENE}"


def _nms_dist(dist):

    return 999.0 if dist is None else dist


def build_dataset():
    print("--- Building the state cache (single GPU) ---")
    print(f"🌍 Target scene: {TARGET_SCENE}")

    scene_suffix = f"_{TARGET_SCENE}" if TARGET_SCENE else "_ALL_SCENES"

    data_ablation_suffix = data_tag()

    tasks = [
        (
            "queries_train",
            f"assets/{VPR_MODEL_ID}/train_trajectories{scene_suffix}{data_ablation_suffix}.json",
        ),
    ]

    pt_memory_cache = {}

    _shard_env = os.getenv("BUILD_SHARD")
    if _shard_env:
        _si, _sn = (int(x) for x in _shard_env.split("/"))
        tasks = [(qf, oj + f".shard{_si}of{_sn}") for qf, oj in tasks]

    for query_folder, output_json in tasks:
        queries_dir = os.path.join(QUERY_DATASET_PATH, query_folder)
        if not os.path.exists(queries_dir):
            print(f"\n⚠️ Skipping {query_folder}: directory does not exist.")
            continue

        query_paths = get_all_images_recursive(queries_dir)
        if _shard_env:
            query_paths = query_paths[_si::_sn]

        _fp = build_fingerprint(query_paths, state_build_settings("train"))
        _why = resume_reason(output_json, _fp, query_paths)
        if _why is None:
            print(f"\n⏩ [Resume] {output_json} matches the current inputs; skipping.")
            continue
        if os.path.exists(output_json):
            print(f"\n♻️ [Rebuild] {output_json}: {_why}")
        if CHECK_ONLY:
            print(f"RESUME_REBUILD: {_why}")
            sys.exit(CHECK_REBUILD_EXIT)

        service = VPRService()
        cropper = service.cropper
        service.model.eval()
        for param in service.model.parameters():
            param.requires_grad = False
        transform = get_base_transform()

        query_paths = get_all_images_recursive(queries_dir)
        if _shard_env:
            query_paths = query_paths[_si::_sn]
        if not query_paths:
            print(f"\n⚠️ No images found under {query_folder}.")
            continue

        dl_data = []
        print(
            f"\n📁 Processing: {query_folder} -> {output_json} (total {len(query_paths)} queries)"
        )

        vpr_core.TARGET_SCENE = TARGET_SCENE

        db_desc, db_rel_paths = service._get_or_compute_db_features(QUERY_DATASET_PATH)
        index = faiss.IndexFlatL2(service.args.fc_output_dim)
        index.add(db_desc)

        db_feat_lookup = {db_rel_paths[i]: db_desc[i] for i in range(len(db_rel_paths))}
        db_rel_paths_list = list(db_rel_paths)

        for q_path in tqdm(
            query_paths,
            desc=f"  ↳ {TARGET_SCENE}:{query_folder} 抽取",
            position=0,
            mininterval=20,
        ):
            q_name = os.path.basename(q_path).split(".")[0]
            q_pose_info = pose_manager.get_pose_info(q_path)
            q_global = q_pose_info["global"] if q_pose_info else None
            if not q_global:
                continue

            query_cv = cv2.imread(q_path)
            img_tensor = (
                transform(Image.open(q_path).convert("RGB"))
                .unsqueeze(0)
                .to(service.args.device)
            )

            with torch.no_grad():
                q_feat_tensor = service.model(img_tensor)
                q_feat_np = q_feat_tensor.cpu().numpy()

            del img_tensor
            del q_feat_tensor

            offline_cache = OfflineQueryMatchCache(
                TARGET_SCENE,
                query_folder,
                q_path,
                cropper,
                service.args.device,
                enabled=True,
            )
            q_feats_lightglue = offline_cache.query_features(query_cv)
            del query_cv

            search_k = min(len(db_rel_paths), TOP_K * 5)
            selected_db_indices, selected_globals, skipped_indices = [], [], []

            q_norm = np.linalg.norm(q_feat_np[0])
            while len(selected_db_indices) < TOP_K:
                distances, predictions = index.search(q_feat_np, search_k)
                selected_db_indices, selected_globals, skipped_indices = [], [], []

                for db_idx in predictions[0]:
                    rel_p = db_rel_paths[db_idx]
                    db_abs_path = os.path.join(QUERY_DATASET_PATH, rel_p)
                    cand_pose_info = pose_manager.get_pose_info(db_abs_path)
                    c_global = cand_pose_info["global"] if cand_pose_info else None

                    if c_global is not None:
                        is_too_close = any(
                            _nms_dist(
                                pose_manager.calculate_physical_distance(
                                    q_path, c_global, sg
                                )
                            )
                            < MIN_CANDIDATE_DIST
                            for sg in selected_globals
                        )
                        if is_too_close:
                            skipped_indices.append(db_idx)
                            continue
                        selected_globals.append(c_global)

                    selected_db_indices.append(db_idx)
                    if len(selected_db_indices) >= TOP_K:
                        break

                if len(selected_db_indices) >= TOP_K or search_k >= len(db_rel_paths):
                    break
                search_k = min(len(db_rel_paths), search_k * 2)

            if len(selected_db_indices) < TOP_K:
                selected_db_indices.extend(
                    skipped_indices[: TOP_K - len(selected_db_indices)]
                )

            for i, db_idx in enumerate(
                tqdm(
                    selected_db_indices,
                    desc=f"    ↳ Process {q_name}",
                    leave=False,
                    position=1,
                    disable=True,
                )
            ):
                rel_p = db_rel_paths[db_idx]
                db_abs_path = os.path.join(QUERY_DATASET_PATH, rel_p)

                asset_path = os.path.join(
                    "assets", "sequence_assets", TARGET_SCENE, rel_p + ".pt"
                )
                if not os.path.exists(asset_path):
                    raise FileNotFoundError(
                        f"Missing sequence asset for retrieved candidate: {asset_path}"
                    )

                try:
                    if asset_path in pt_memory_cache:
                        seq_data = pt_memory_cache[asset_path]
                    else:
                        seq_data = torch.load(
                            asset_path, map_location="cpu", weights_only=False
                        )
                        if len(pt_memory_cache) >= 100:
                            oldest_key = next(iter(pt_memory_cache))
                            del pt_memory_cache[oldest_key]
                        pt_memory_cache[asset_path] = seq_data

                    asset_error = validate_sequence_asset(seq_data, rel_p, TARGET_SCENE)

                    if asset_error:

                        raise RuntimeError(
                            f"Invalid sequence asset {asset_path}: {asset_error}"
                        )

                    lg_seq_feats = seq_data.get(
                        "lightfeats", seq_data.get("lightglue_feats", [])
                    )
                    rel_seq_paths = seq_data.get("rel_seq_paths", [])
                except Exception as exc:
                    raise RuntimeError(
                        f"Could not load sequence asset: {asset_path}: {exc}"
                    ) from exc

                if not lg_seq_feats or len(lg_seq_feats) == 0:
                    raise RuntimeError(
                        f"Sequence asset has no local features: {asset_path}"
                    )

                fmt = seq_data.get("format", SEQUENCE_ASSET_FORMAT)
                if fmt != SEQUENCE_ASSET_FORMAT:
                    raise RuntimeError(
                        f"Sequence asset {asset_path} has payload format "
                        f"{fmt!r}, expected {SEQUENCE_ASSET_FORMAT!r}. Rebuild "
                        "with FORCE_REBUILD_SEQUENCE_ASSETS=1."
                    )

                h_paths, f_paths = get_database_spatial_sequence(
                    db_abs_path, pose_manager, SPATIAL_STEPS
                )
                rel_seq_paths = [
                    os.path.relpath(path, QUERY_DATASET_PATH)
                    for path in list(reversed(h_paths)) + [db_abs_path] + f_paths
                ]

                seq_len = len(rel_seq_paths)
                center_idx = seq_len // 2

                if DATA_ABLATION_RANDOM_WINDOW:
                    query_rng = random.Random(
                        f"window-v2|{TARGET_SCENE}|{q_name}|{rel_p}"
                    )
                    random_paths = query_rng.sample(db_rel_paths_list, seq_len)
                    random_paths[center_idx] = rel_p
                    new_rel_seq_paths = random_paths
                    new_lg_seq_feats = []

                    for rp in new_rel_seq_paths:
                        rp_asset = os.path.join(
                            "assets", "sequence_assets", TARGET_SCENE, rp + ".pt"
                        )
                        try:
                            if rp_asset in pt_memory_cache:
                                rp_data = pt_memory_cache[rp_asset]
                            else:
                                rp_data = torch.load(
                                    rp_asset, map_location="cpu", weights_only=False
                                )
                                if len(pt_memory_cache) >= 100:
                                    del pt_memory_cache[next(iter(pt_memory_cache))]
                                pt_memory_cache[rp_asset] = rp_data

                            asset_error = validate_sequence_asset(
                                rp_data, rp, TARGET_SCENE
                            )

                            if asset_error:

                                raise RuntimeError(
                                    f"Invalid sequence asset {rp_asset}: {asset_error}"
                                )

                            rp_lg_feats = rp_data.get(
                                "lightfeats", rp_data.get("lightglue_feats", [])
                            )
                            rp_paths = rp_data.get("rel_seq_paths", [])
                            if rp_paths and len(rp_lg_feats) > 0:
                                rp_center_idx = len(rp_paths) // 2
                                new_lg_seq_feats.append(rp_lg_feats[rp_center_idx])
                            else:
                                raise RuntimeError(
                                    f"Sequence asset has no centre feature: {rp_asset}"
                                )
                        except Exception as exc:
                            raise RuntimeError(
                                f"Could not load random-window asset: {rp_asset}: {exc}"
                            ) from exc

                    rel_seq_paths = new_rel_seq_paths
                    lg_seq_feats = new_lg_seq_feats

                else:
                    new_lg_seq_feats = []
                    for rp in rel_seq_paths:
                        rp_asset = os.path.join(
                            "assets", "sequence_assets", TARGET_SCENE, rp + ".pt"
                        )
                        try:
                            if rp_asset in pt_memory_cache:
                                rp_data = pt_memory_cache[rp_asset]
                            else:
                                rp_data = torch.load(
                                    rp_asset, map_location="cpu", weights_only=False
                                )
                                if len(pt_memory_cache) >= 100:
                                    del pt_memory_cache[next(iter(pt_memory_cache))]
                                pt_memory_cache[rp_asset] = rp_data

                            asset_error = validate_sequence_asset(
                                rp_data, rp, TARGET_SCENE
                            )

                            if asset_error:

                                raise RuntimeError(
                                    f"Invalid sequence asset {rp_asset}: {asset_error}"
                                )

                            rp_lg_feats = rp_data.get(
                                "lightfeats", rp_data.get("lightglue_feats", [])
                            )
                            rp_paths = rp_data.get("rel_seq_paths", [])
                            if rp_paths and len(rp_lg_feats) > 0:
                                rp_center_idx = len(rp_paths) // 2
                                new_lg_seq_feats.append(rp_lg_feats[rp_center_idx])
                            else:
                                raise RuntimeError(
                                    f"Sequence asset has no centre feature: {rp_asset}"
                                )
                        except Exception as exc:
                            raise RuntimeError(
                                f"Could not load no-ROI asset: {rp_asset}: {exc}"
                            ) from exc

                    lg_seq_feats = new_lg_seq_feats

                step_errors_seq, step_scores_seq = [], []
                for rel_sp in rel_seq_paths:
                    sp_abs = os.path.join(QUERY_DATASET_PATH, rel_sp)
                    sp_info = pose_manager.get_pose_info(sp_abs)
                    sp_g = sp_info["global"] if sp_info else None
                    dist = (
                        pose_manager.calculate_physical_distance(q_path, q_global, sp_g)
                        if sp_g
                        else None
                    )
                    step_errors_seq.append(round(dist, 3) if dist is not None else -1.0)

                    sp_feat = db_feat_lookup.get(rel_sp)
                    if sp_feat is not None:
                        cos_sim = np.dot(sp_feat, q_feat_np[0]) / (
                            np.linalg.norm(sp_feat) * q_norm
                        )
                        step_scores_seq.append(round(max(0.0, float(cos_sim)), 4))
                    else:
                        step_scores_seq.append(0.0)

                geo_seq = {k: [] for k in GEOMETRY_KEYS}
                inliers_seq, area_step_seq, area_query_seq, scale_ratio_seq = (
                    [],
                    [],
                    [],
                    [],
                )

                for step_idx in range(len(lg_seq_feats)):
                    valid_inliers, kp_area_s, kp_area_q, scale_ratio = 0, 0.0, 0.0, 0.0
                    geo = dict.fromkeys(GEOMETRY_KEYS, 0.0)

                    if lg_seq_feats[step_idx]:
                        step_feats_gpu = cropper.unpack_offline_features(
                            lg_seq_feats[step_idx], service.args.device
                        )
                        valid_inliers, scale_ratio, kp_area_s, kp_area_q, geo = (
                            offline_cache.geometric_metrics(
                                rel_seq_paths[step_idx],
                                os.path.join(
                                    "assets",
                                    "sequence_assets",
                                    TARGET_SCENE,
                                    rel_seq_paths[step_idx] + ".pt",
                                ),
                                q_feats_lightglue,
                                step_feats_gpu,
                            )
                        )
                        del step_feats_gpu

                    inliers_seq.append(valid_inliers)
                    area_step_seq.append(round(kp_area_s, 2))
                    area_query_seq.append(round(kp_area_q, 2))
                    scale_ratio_seq.append(round(scale_ratio, 4))
                    for _gk in GEOMETRY_KEYS:
                        geo_seq[_gk].append(round(float(geo[_gk]), 5))

                if inliers_seq:
                    dl_data.append(
                        {
                            "query": q_name,
                            "rank": i + 1,
                            "candidate_name": os.path.basename(rel_p),
                            "error": (
                                round(step_errors_seq[len(step_errors_seq) // 2], 3)
                                if step_errors_seq
                                else -1.0
                            ),
                            "scores": step_scores_seq,
                            "step_errors": step_errors_seq,
                            "inliers": inliers_seq,
                            "area_step": area_step_seq,
                            "area_query": area_query_seq,
                            "scale_ratio": scale_ratio_seq,
                            **{k: geo_seq[k] for k in GEOMETRY_KEYS},
                        }
                    )
            offline_cache.flush()
            del q_feats_lightglue

        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        temporary = f"{output_json}.tmp.{os.getpid()}"
        try:
            with open(temporary, "w", encoding="utf-8") as f:
                json.dump(dl_data, f, indent=2, ensure_ascii=False)
            os.replace(temporary, output_json)
        finally:
            if os.path.exists(temporary):
                os.remove(temporary)
        write_fingerprint(
            output_json,
            build_fingerprint(query_paths, state_build_settings("train")),
        )

        true_count = sum(
            1 for d in dl_data if 0 <= d.get("error", -1) <= SUCCESS_THRESHOLD_TRAIN
        )

        if not dl_data:
            raise SystemExit(
                f"[FATAL] {query_folder}: extracted 0 candidate sequences. "
                "Every observed cause was silent and upstream: missing "
                "assets/sequence_assets/<Scene>/ (step 2 not run), stale "
                "empty *_trajectories json, or pose lookups returning None "
                "(missing pose marker file in the recording). Fix the input; "
                "a green run with 0 rows only detonates later in training "
                "as num_samples=0."
            )

        print(
            f"✅ {query_folder} done! Extracted {len(dl_data)} candidate sequences; true hits: {true_count}."
        )


if __name__ == "__main__":
    build_dataset()
