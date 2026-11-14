import os
import sys
import json
import torch
import numpy as np
import csv
from collections import defaultdict
import time
from datetime import datetime
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import (
    data_tag,
    TARGET_SCENE,
    SUCCESS_THRESHOLD_TEST,
    SUCCESS_THRESHOLD_TRAIN,
    TOP_K,
    DL_MODEL_SAVE_PATH as BASE_MODEL_SAVE_PATH,
    agent_input_dim,
    agent_tag,
    DL_HIDDEN_DIM,
    SPATIAL_STEPS,
    INLIER_NORM_FACTOR,
    VPR_MODEL_TYPE,
    VPR_MODEL_ID,
    LOG1P,
    ABLATION_USE_AREA,
    ABLATION_USE_DERIVATIVES,
    ABLATION_USE_PROGRESS,
    ABLATION_USE_SIM,
    ABLATION_USE_RANK,
    ABLATION_USE_DINO,
    ABLATION_USE_POSE,
    ABLATION_USE_HULL,
    ABLATION_USE_HOMOG,
    ABLATION_USE_WALK,
    ABLATION_USE_SHAPE,
    ABLATION_USE_ZOOM,
    ABLATION_USE_ZOOMSHUF,
    ABLATION_USE_CORE,
    ABLATION_USE_SCALE,
    ABLATION_INPUT_NORM,
    ABLATION_ARCH,
    DATA_ABLATION_RANDOM_WINDOW,
)

OUTPUT_RAW_DATA = False

GATE_MARGIN = float(os.getenv("STAR_GATE_MARGIN", "0"))

SUE_SLOPES = (3.5, 35.0, 350.0, 3500.0)

from sl_agent.vpr_classifier import (
    VPRReranking,
    build_extra_channels,
    build_sequence_features,
    center_frame_view,
    head_steps,
)
from core.paper_config import load_paper_config


def main():
    print("\n" + "=" * 80)
    print("[Fast Eval] Starting Offline Evaluation Engine")
    print("=" * 80)

    dynamic_input_dim = agent_input_dim()
    ablation_str = agent_tag()

    data_ablation_suffix = data_tag()

    weight_scene = os.getenv("STAR_WEIGHT_SCENE", TARGET_SCENE)
    if weight_scene != TARGET_SCENE:
        print(f"[Info] Weight scene override: {weight_scene} (data: {TARGET_SCENE})")
    scene_suffix = f"_{weight_scene}" if weight_scene else "_ALL_SCENES"
    default_seed = int(load_paper_config()["seeds"][0])
    _seed = int(os.getenv("TRAIN_SEED", str(default_seed)))
    seed_suffix = f"_seed{_seed}" if _seed != default_seed else ""
    if os.getenv("ENABLE_SCENE_MODELS_CACHE") == "True":
        model_save_path = BASE_MODEL_SAVE_PATH.replace(
            ".pth",
            f"_{ablation_str}{scene_suffix}{data_ablation_suffix}{seed_suffix}.pth",
        )
    else:
        model_save_path = BASE_MODEL_SAVE_PATH.replace(
            ".pth", f"_{ablation_str}{data_ablation_suffix}{seed_suffix}.pth"
        )

    if os.getenv("USE_MASTER_WEIGHT") == "True":
        model_save_path = model_save_path.replace(VPR_MODEL_ID, "MegaLoc_8448")
        print(f"[Info] Cross-backbone zero-shot transfer activated.")
        print(f"[Info] Redirecting Teacher weights -> {model_save_path}")

    print(f"[Config] Dynamic Feature Dim: {dynamic_input_dim}D")
    print(f"[Config] Model Architecture: {ABLATION_ARCH.upper()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VPRReranking(
        input_dim=dynamic_input_dim,
        hidden_dim=DL_HIDDEN_DIM,
        arch=ABLATION_ARCH,
        input_norm=ABLATION_INPUT_NORM,
    ).to(device)

    if not os.path.exists(model_save_path):
        raise FileNotFoundError(
            f"Pre-trained ablation model not found: {model_save_path}. "
            "Run train_agent.py with the same config first."
        )

    model.load_state_dict(
        torch.load(model_save_path, map_location=device, weights_only=True)
    )
    model.eval()
    print(f"[Success] Loaded model weights: {model_save_path}")

    scene_suffix = f"_{TARGET_SCENE}" if TARGET_SCENE else "_ALL_SCENES"

    print(
        f"[Status] Data Ablation: {data_ablation_suffix if data_ablation_suffix else 'Normal (Full Data)'}"
    )

    json_path = f"assets/{VPR_MODEL_ID}/test_trajectories{scene_suffix}{data_ablation_suffix}.json"
    if len(sys.argv) > 1:
        json_path = sys.argv[1]

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Test asset not found: {json_path}")

    expected_steps = sum(SPATIAL_STEPS) + 1

    with open(json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    from train_agent import _require_temporal_rows

    _require_temporal_rows(raw_data, json_path, expected_steps)

    queries_dict = defaultdict(list)
    for item in raw_data:
        queries_dict[item["query"]].append(item)

    total_queries = len(queries_dict)
    print(
        f"[Success] Test assets loaded: {json_path} (Total {total_queries} Queries)\n"
    )

    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(os.path.join("logs", "results"), exist_ok=True)
    report_csv_path = os.path.join(
        "logs",
        "results",
        f"fast_eval_{ablation_str}{data_ablation_suffix}_{timestamp}.csv",
    )

    if OUTPUT_RAW_DATA:
        fieldnames = [
            "rank",
            "is_success_test",
            "is_success_train",
            "distance_error",
            "query_image",
            "matched_image",
        ]
        with open(report_csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    raw_success_test, sgv_success_test, score_smooth_success_test = 0, 0, 0
    ours_success_test = 0
    selective_gv_data_test = []

    raw_success_train, sgv_success_train, score_smooth_success_train = 0, 0, 0
    ours_success_train = 0
    selective_gv_data_train = []

    all_distance_errors = []
    total_processed = 0
    max_steps = sum(SPATIAL_STEPS) + 1
    start_time = time.time()
    head_window = head_steps(ABLATION_ARCH, max_steps)
    if head_window != max_steps:
        print(
            f"[Config] Head window: {head_window}/{max_steps} steps (centre frame only)"
        )

    dummy = build_sequence_features(
        [],
        [],
        head_window,
        use_derivatives=ABLATION_USE_DERIVATIVES,
        use_progress=ABLATION_USE_PROGRESS,
        use_core=ABLATION_USE_CORE,
        use_scale=ABLATION_USE_SCALE,
    )

    for q_name, cands in queries_dict.items():
        total_processed += 1
        cands = sorted(cands, key=lambda x: x.get("rank", 999))[:TOP_K]
        if not cands:
            continue

        valid_errors = [
            c.get("error", -1.0) for c in cands if c.get("error", -1.0) >= 0
        ]

        is_raw_correct_test = (
            1 if (valid_errors and valid_errors[0] <= SUCCESS_THRESHOLD_TEST) else 0
        )
        is_raw_correct_train = (
            1 if (valid_errors and valid_errors[0] <= SUCCESS_THRESHOLD_TRAIN) else 0
        )
        raw_success_test += is_raw_correct_test
        raw_success_train += is_raw_correct_train

        best_sgv_cand = max(
            cands,
            key=lambda x: (
                x["inliers"][len(x["inliers"]) // 2] if x.get("inliers") else 0
            ),
        )
        is_sgv_correct_test = (
            1
            if (0 <= best_sgv_cand.get("error", -1.0) <= SUCCESS_THRESHOLD_TEST)
            else 0
        )
        is_sgv_correct_train = (
            1
            if (0 <= best_sgv_cand.get("error", -1.0) <= SUCCESS_THRESHOLD_TRAIN)
            else 0
        )
        sgv_success_test += is_sgv_correct_test
        sgv_success_train += is_sgv_correct_train

        def _center_sim(c):
            s = c.get("scores")
            return abs(s[len(s) // 2]) if s else 0.0

        dists = [max(0.0, 1.0 - _center_sim(c)) for c in cands]
        pa_conf = -1.0 * (dists[0] / (dists[1] + 1e-6)) if len(dists) >= 2 else -1.0

        sue_cands = [c for c in cands if "global_pose" in c]
        if len(sue_cands) >= 2:
            num_NN = min(10, len(sue_cands))
            poses = np.array([c["global_pose"][:2] for c in sue_cands[:num_NN]])
            d_l2 = np.array(
                [
                    math.sqrt(max(0.0, 2.0 - 2.0 * _center_sim(c)))
                    for c in sue_cands[:num_NN]
                ]
            )
            sue_confs = {}
            for slope in SUE_SLOPES:
                w = np.exp(-slope * (d_l2 - d_l2.min()))
                w = w / (np.sum(w) + 1e-8)
                mean_pose = np.array(
                    [
                        np.average(poses[:, 0], weights=w),
                        np.average(poses[:, 1], weights=w),
                    ]
                )
                var_lat = sum(
                    w[k] * (min(500.0, abs(poses[k, 0] - mean_pose[0])) ** 2)
                    for k in range(num_NN)
                )
                var_lon = sum(
                    w[k] * (min(500.0, abs(poses[k, 1] - mean_pose[1])) ** 2)
                    for k in range(num_NN)
                )
                sue_confs[slope] = -1.0 * (float(var_lat) + float(var_lon)) / 2.0
        else:
            sue_confs = {slope: -float("inf") for slope in SUE_SLOPES}

        inlier_conf = float(
            max(cands[0].get("inliers", [0]))
            if cands and cands[0].get("inliers")
            else 0.0
        )

        sue_keys = {f"sue_{slope}": sue_confs[slope] for slope in SUE_SLOPES}
        selective_gv_data_test.append(
            {
                "raw": is_raw_correct_test,
                "sgv": is_sgv_correct_test,
                "pa": pa_conf,
                "inliers": inlier_conf,
                **sue_keys,
            }
        )
        selective_gv_data_train.append(
            {
                "raw": is_raw_correct_train,
                "sgv": is_sgv_correct_train,
                "pa": pa_conf,
                "inliers": inlier_conf,
                **sue_keys,
            }
        )

        def _valid_mean_score(c):
            s = c.get("scores")
            if not s:
                return 0.0
            valid = c.get("step_valid") or [1.0] * len(s)
            vs = [x for x, v in zip(s, valid) if v > 0.5]
            if not vs:
                vs = [s[len(s) // 2]]
            return sum(vs) / len(vs)

        best_score_smooth_cand = max(cands, key=_valid_mean_score)
        if 0 <= best_score_smooth_cand.get("error", -1.0) <= SUCCESS_THRESHOLD_TEST:
            score_smooth_success_test += 1
        if 0 <= best_score_smooth_cand.get("error", -1.0) <= SUCCESS_THRESHOLD_TRAIN:
            score_smooth_success_train += 1

        all_features = []
        valid_cands = []
        head_cands = (
            center_frame_view(cands, max_steps) if head_window != max_steps else cands
        )
        extra = build_extra_channels(
            head_cands,
            head_window,
            ABLATION_USE_SIM,
            ABLATION_USE_AREA,
            ABLATION_USE_RANK,
            ABLATION_USE_DINO,
            ABLATION_USE_POSE,
            ABLATION_USE_HULL,
            ABLATION_USE_HOMOG,
            ABLATION_USE_WALK,
            ABLATION_USE_SHAPE,
            ABLATION_USE_ZOOM,
            ABLATION_USE_ZOOMSHUF,
        )
        n_extra = len(extra[0]) if extra else 0
        for ci, cand in enumerate(cands):
            valid_cands.append(cand)
            inliers = head_cands[ci].get("inliers", [])
            scales = head_cands[ci].get("scale_ratio", [])
            cand_feat = build_sequence_features(
                inliers,
                scales,
                head_window,
                use_derivatives=ABLATION_USE_DERIVATIVES,
                use_progress=ABLATION_USE_PROGRESS,
                extra_channels=extra[ci] if extra else None,
                use_core=ABLATION_USE_CORE,
                use_scale=ABLATION_USE_SCALE,
            )
            all_features.append(cand_feat)

        pad_steps = [row + [0.0] * n_extra for row in dummy]
        while len(all_features) < TOP_K:
            all_features.append(pad_steps)

        x_tensor = torch.tensor([all_features], dtype=torch.float32).to(device)
        mask_list = [1.0] * len(valid_cands) + [0.0] * (TOP_K - len(valid_cands))
        mask_tensor = torch.tensor([mask_list], dtype=torch.float32).to(device)

        best_match_candidate = valid_cands[0] if valid_cands else None
        if best_match_candidate:
            for t in range(1, head_window + 1):
                step_x = x_tensor[:, :, :t, :]
                with torch.no_grad():
                    probs = model.predict_softmax(step_x, mask=mask_tensor)
                    max_p, max_idx = torch.max(probs[0, :, -1], dim=0)

            with torch.no_grad():
                probs = model.predict_softmax(x_tensor, mask=mask_tensor)
                max_p, max_idx = torch.max(probs[0, :, -1], dim=0)
            if max_idx.item() < len(valid_cands):
                margin = float(probs[0, max_idx, -1] - probs[0, 0, -1])
                if margin >= GATE_MARGIN:
                    best_match_candidate = valid_cands[max_idx.item()]

        distance_error = (
            best_match_candidate.get("error", None) if best_match_candidate else None
        )
        matched_name = (
            best_match_candidate.get("candidate_name", "Unknown")
            if best_match_candidate
            else "Unknown"
        )
        rank = best_match_candidate.get("rank", -1) if best_match_candidate else -1

        is_success_test = False
        is_success_train = False

        if distance_error is not None and distance_error >= 0:
            all_distance_errors.append(distance_error)
            if distance_error <= SUCCESS_THRESHOLD_TEST:
                is_success_test = True
                ours_success_test += 1
            if distance_error <= SUCCESS_THRESHOLD_TRAIN:
                is_success_train = True
                ours_success_train += 1

        if OUTPUT_RAW_DATA:
            with open(report_csv_path, "a", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(
                    {
                        "rank": rank,
                        "is_success_test": is_success_test,
                        "is_success_train": is_success_train,
                        "distance_error": (
                            distance_error
                            if distance_error is not None and distance_error >= 0
                            else ""
                        ),
                        "query_image": q_name,
                        "matched_image": matched_name,
                    }
                )

        dist_str = (
            f"{distance_error:.2f}"
            if (distance_error is not None and distance_error >= 0)
            else "N/A"
        )
        print(
            f"  └─ [{total_processed}/{total_queries}] {q_name} -> Target: {matched_name} | Error: {dist_str}m | Success(Test/Train): {is_success_test}/{is_success_train}"
        )

    total_time = time.time() - start_time

    def evaluate_selective_gv(name, key, data, prefix=""):
        if not data:
            return 0
        best_s, best_th, best_raw_count, best_sgv_count = -1, None, 0, 0
        ths = sorted(list(set(d[key] for d in data)))
        print(
            f"\n[Selective GV Analysis] Finding optimal threshold for {name} [{prefix}] ({len(ths)} candidates)..."
        )

        for th in ths:
            s, raw_count, sgv_count = 0, 0, 0
            for d in data:
                if d[key] >= th:
                    s += d["raw"]
                    raw_count += 1
                else:
                    s += d["sgv"]
                    sgv_count += 1

            if s > best_s:
                best_s, best_th, best_raw_count, best_sgv_count = (
                    s,
                    th,
                    raw_count,
                    sgv_count,
                )

        print(f"  - Optimal Threshold = {best_th:.4f}")
        print(
            f"  - Strategy Dist: Trust Global(Raw) {best_raw_count} times | Switch to Local(SGV) {best_sgv_count} times"
        )
        print(f"  - Final Score: {best_s}/{len(data)} ({(best_s/len(data))*100:.2f}%)")

        if best_sgv_count == 0:
            print(
                "  [Insight] Model trusted Raw 100%. Triggering SGV would lower the score, showing limitations of passive defense."
            )
        return best_s

    sue_best_success_test = max(
        evaluate_selective_gv(
            f"SUE-Score (CVPR'24) slope={slope}",
            f"sue_{slope}",
            selective_gv_data_test,
            f"TEST={SUCCESS_THRESHOLD_TEST}m",
        )
        for slope in SUE_SLOPES
    )

    sue_best_success_train = max(
        evaluate_selective_gv(
            f"SUE-Score (CVPR'24) slope={slope}",
            f"sue_{slope}",
            selective_gv_data_train,
            f"TRAIN={SUCCESS_THRESHOLD_TRAIN}m",
        )
        for slope in SUE_SLOPES
    )

    def print_and_save_report(
        prefix,
        threshold,
        raw_succ,
        sgv_succ,
        smooth_succ,
        sue_succ,
        ours_succ,
    ):
        print(
            f"\n\n[Evaluation Complete] Paper Report - [{prefix} THRESHOLD = {threshold}m]"
        )
        print("-" * 80)
        print(
            f"| {'Method (Re-ranking Strategy)':<33} | {'Success':<7} | {'Accuracy':<8} | {'Note'}"
        )
        print("-" * 80)

        def print_row(name, success, story):
            acc = (success / total_queries) * 100 if total_queries > 0 else 0
            print(
                f"| {name:<33} | {success:>3}/{total_queries:<3} | {acc:>7.2f}% | {story}"
            )

        print_row(
            f"Raw VPR Base: {VPR_MODEL_TYPE}", raw_succ, "Baseline: Raw VPR Output"
        )
        print_row("Naive Temporal Smoothing", smooth_succ, "Baseline: SeqSLAM Degraded")
        print_row(
            "Single-Frame Dense GV (LightGlue)", sgv_succ, "Baseline: Max Match Points"
        )
        print("-" * 80)

        print_row("SUE-Selective GV", sue_succ, "CVPR 2024")
        print("-" * 80)

        display_ablation = f"arch={ABLATION_ARCH}, deriv={ABLATION_USE_DERIVATIVES}, prog={ABLATION_USE_PROGRESS}"
        if DATA_ABLATION_RANDOM_WINDOW:
            display_ablation += ", random_window"

        ours_name_str = f"Ours ({display_ablation})"

        print(
            f"| \033[1;36m{ours_name_str:<33}\033[0m | \033[1;36m{ours_succ:>3}/{total_queries:<3}\033[0m | \033[1;36m{(ours_succ/total_queries)*100:>7.2f}%\033[0m | \033[1;36mCurrent Ablation Results\033[0m"
        )
        print("-" * 80)

        if total_processed > 0:
            with open(report_csv_path, "a", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([])
                writer.writerow(
                    [
                        "=" * 10,
                        f"TABLE RESULTS (TOP-1 ACC) - {prefix} THRESHOLD = {threshold}m",
                        "=" * 10,
                        "",
                        "",
                    ]
                )
                writer.writerow([f"SCENE:{TARGET_SCENE},  Baseline:{VPR_MODEL_TYPE}"])
                writer.writerow(
                    [
                        "Raw VPR Base",
                        f"{(raw_succ/total_queries)*100:.2f}%",
                        f"{raw_succ}/{total_queries}",
                        "",
                        "",
                    ]
                )
                writer.writerow(
                    [
                        "Naive Temporal Smoothing",
                        f"{(smooth_succ/total_queries)*100:.2f}%",
                        f"{smooth_succ}/{total_queries}",
                        "SeqSLAM Degraded",
                        "",
                    ]
                )
                writer.writerow(
                    [
                        "Single-Frame Dense GV",
                        f"{(sgv_succ/total_queries)*100:.2f}%",
                        f"{sgv_succ}/{total_queries}",
                        "SuperPoint+LightGlue",
                        "",
                    ]
                )
                writer.writerow(
                    [
                        "SUE-Selective GV",
                        f"{(sue_succ/total_queries)*100:.2f}%",
                        f"{sue_succ}/{total_queries}",
                        "CVPR 2024",
                        "",
                    ]
                )
                writer.writerow(
                    [
                        ours_name_str,
                        f"{(ours_succ/total_queries)*100:.2f}%",
                        f"{ours_succ}/{total_queries}",
                        "Ablation Version",
                        "",
                    ]
                )

    print_and_save_report(
        "TRAIN",
        SUCCESS_THRESHOLD_TRAIN,
        raw_success_train,
        sgv_success_train,
        score_smooth_success_train,
        sue_best_success_train,
        ours_success_train,
    )
    print_and_save_report(
        "TEST",
        SUCCESS_THRESHOLD_TEST,
        raw_success_test,
        sgv_success_test,
        score_smooth_success_test,
        sue_best_success_test,
        ours_success_test,
    )

    print(f"\n[Info] Core metrics saved to: {report_csv_path}")


if __name__ == "__main__":
    main()
