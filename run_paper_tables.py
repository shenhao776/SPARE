import argparse
from collections import OrderedDict, defaultdict
from datetime import datetime
import json
import math
import os
import statistics
import sys
import time

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

_ROOT = os.path.dirname(os.path.abspath(__file__))
if not os.path.isdir(os.path.join(_ROOT, "configs")):
    _ROOT = os.path.dirname(_ROOT)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_PIPELINE = os.path.join(_ROOT, "pipeline")
if os.path.isdir(_PIPELINE) and _PIPELINE not in sys.path:
    sys.path.insert(1, _PIPELINE)

from core.env_isolation import (
    ensure_reference_scene,
    scrub_environment,
)

_SCRUBBED = scrub_environment()

ensure_reference_scene()

from core.config import (
    DL_HIDDEN_DIM,
    data_tag,
    agent_input_dim,
    agent_tag,
    INLIER_NORM_FACTOR,
    LOG1P,
    paper_protocol_env,
    SPATIAL_STEPS,
    TOP_K,
)
from core.paper_config import format_path, load_paper_config
from sl_agent.vpr_classifier import (
    VPRReranking,
    build_extra_channels,
    build_sequence_features,
    center_frame_view,
    head_steps,
)

FULL_VARIANT = paper_protocol_env()
PAPER_CONFIG = load_paper_config(FULL_VARIANT)
OWN_DATASETS = {
    name: list(spec["scenes"]) for name, spec in PAPER_CONFIG["datasets"].items()
}
PAPER_DATASETS = list(OWN_DATASETS)
PLACEHOLDER_DATASETS = set(PAPER_CONFIG["placeholders"])
PAPER_BACKBONES = list(PAPER_CONFIG["backbones"])
MODEL_IDS = dict(PAPER_CONFIG["model_ids"])
OWN_TEST_THRESHOLDS = dict(PAPER_CONFIG["scene_thresholds"])
MAIN_BACKBONE = PAPER_CONFIG["main_backbone"]
AGENT_BACKBONE = PAPER_CONFIG["agent_backbone"]
PAPER_SEEDS = tuple(int(seed) for seed in PAPER_CONFIG["seeds"])
ABLATION_VARIANTS = [
    (row["label"], row.get("variant")) for row in PAPER_CONFIG["ablation_rows"]
]
PROSE_VARIANTS = [
    (row["label"], row["variant"]) for row in PAPER_CONFIG["prose_variants"]
]

MAX_STEPS = sum(SPATIAL_STEPS) + 1
_SCENE_STEPS = {}


def _scene_max_steps(scene):

    if scene not in _SCENE_STEPS:
        from core.config import resolve_dataset

        try:
            steps = resolve_dataset(scene).get("sequence", {}).get("spatial_steps")
        except FileNotFoundError:
            steps = None
        _SCENE_STEPS[scene] = (sum(steps) + 1) if steps is not None else MAX_STEPS
    return _SCENE_STEPS[scene]


SUE_SLOPES = (3.5, 35.0, 350.0, 3500.0)
OWN_METRIC_CACHE = {}
_RECORDS_CACHE = OrderedDict()
_RECORDS_CACHE_SLOTS = 3
_FEATURES_CACHE = OrderedDict()
_FEATURES_CACHE_SLOTS = 3
OWN_AGENT_CACHE = {}
FORCE_CPU = False
MISSING_CELL = PAPER_CONFIG["missing_cell"]
MISSING_ITEMS = {}


def _csv(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def _name_aliases(names):

    aliases = {}
    for name in names:
        lower = name.lower()
        aliases[lower] = name
        aliases["".join(ch for ch in lower if ch.isalnum())] = name
    return aliases


def _dataset_aliases():
    aliases = _name_aliases(PAPER_DATASETS)
    for name, spec in PAPER_CONFIG["datasets"].items():
        for alias in spec.get("aliases", []):
            aliases[str(alias).lower()] = name
    return aliases


def _record_missing(context, reason, log_path=None):

    message = str(reason).replace("\n", " ")
    MISSING_ITEMS.setdefault(context, message)
    if log_path:
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(f"MISSING {context}: {message}\n")


def _parse_named_list(value, aliases, valid, parser, flag):
    selected = []
    for item in _csv(value):
        name = aliases.get(item.lower(), item)
        if name not in valid:
            parser.error(
                f"unknown value {item!r} for {flag}; valid: {', '.join(valid)}"
            )
        if name not in selected:
            selected.append(name)
    return selected


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Replay all SPARE paper tables directly from assets/."
    )
    parser.add_argument(
        "--tables",
        default="all",
        help="comma-separated: main,backbones,ablation (default: all)",
    )
    parser.add_argument(
        "--datasets",
        default=",".join(PAPER_DATASETS),
        help="comma-separated paper datasets (default: all configured paper datasets)",
    )
    parser.add_argument(
        "--backbones",
        default=",".join(PAPER_BACKBONES),
        help="comma-separated VPR backbones (default: all configured backbones)",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="diagnostic CPU fallback (the exact replay runs on the GPU)",
    )
    args = parser.parse_args()

    table_aliases = {
        "all": "all",
        "main": "main",
        "table2": "main",
        "backbones": "backbones",
        "table3": "backbones",
        "backbones_retrained": "backbones_retrained",
        "table3_retrained": "backbones_retrained",
        "ablation": "ablation",
        "table4": "ablation",
        "threshold25": "threshold25",
        "table5": "threshold25",
        "crop_sweep": "crop_sweep",
        "table6": "crop_sweep",
        "crossmap": "crossmap",
        "cross_map": "crossmap",
        "fixedhead": "fixedhead",
        "fixed_head": "fixedhead",
    }
    tables = _parse_named_list(
        args.tables,
        table_aliases,
        [
            "all",
            "main",
            "backbones",
            "backbones_retrained",
            "ablation",
            "threshold25",
            "crop_sweep",
            "crossmap",
            "fixedhead",
        ],
        parser,
        "--tables",
    )
    args.tables = (
        ["main", "backbones", "ablation", "threshold25", "crossmap", "fixedhead"]
        if "all" in tables
        else tables
    )

    dataset_aliases = _dataset_aliases()
    args.datasets = _parse_named_list(
        args.datasets,
        dataset_aliases,
        PAPER_DATASETS,
        parser,
        "--datasets",
    )
    backbone_aliases = _name_aliases(PAPER_BACKBONES)
    args.backbones = _parse_named_list(
        args.backbones,
        backbone_aliases,
        PAPER_BACKBONES,
        parser,
        "--backbones",
    )
    return args


def _variant_channels(variant):

    return {
        kw: variant[env_name] == "True"
        for kw, env_name in (
            ("use_sim", "ABLATION_USE_SIM"),
            ("use_area", "ABLATION_USE_AREA"),
            ("use_rank", "ABLATION_USE_RANK"),
            ("use_dino", "ABLATION_USE_DINO"),
            ("use_pose", "ABLATION_USE_POSE"),
            ("use_hull", "ABLATION_USE_HULL"),
            ("use_homog", "ABLATION_USE_HOMOG"),
            ("use_walk", "ABLATION_USE_WALK"),
            ("use_shape", "ABLATION_USE_SHAPE"),
            ("use_zoom", "ABLATION_USE_ZOOM"),
            ("use_zoomshuf", "ABLATION_USE_ZOOMSHUF"),
            ("use_vprpatch", "ABLATION_USE_VPRPATCH"),
            ("use_vprpatchshuf", "ABLATION_USE_VPRPATCHSHUF"),
        )
    }


def _own_agent(
    scene,
    variant,
    device,
    seed=None,
    threshold=None,
    trained_backbone=AGENT_BACKBONE,
    weight_scene=None,
    weight_threshold=None,
):
    scene = weight_scene or scene
    arch = variant["ABLATION_ARCH"]
    derivatives = variant["ABLATION_USE_DERIVATIVES"] == "True"
    progress = variant["ABLATION_USE_PROGRESS"] == "True"
    core = variant["ABLATION_USE_CORE"] == "True"
    scale = variant["ABLATION_USE_SCALE"] == "True"
    innorm = variant["ABLATION_INPUT_NORM"] == "True"
    channels = _variant_channels(variant)
    suffix = data_tag(
        random_window=variant["DATA_ABLATION_RANDOM_WINDOW"] == "True",
    )
    th_tag = ""
    head_th = weight_threshold if weight_threshold is not None else threshold
    deploy_th = OWN_TEST_THRESHOLDS.get(scene, float(REPORT_THRESHOLDS[0]))
    if head_th is not None and abs(head_th - deploy_th) > 1e-9:
        th_tag = f"_th{head_th:g}"

    seed_tag = f"_seed{seed}" if seed is not None else ""
    key = (
        scene,
        trained_backbone,
        arch,
        derivatives,
        progress,
        core,
        scale,
        innorm,
        tuple(sorted(channels.items())),
        suffix,
        seed_tag,
        th_tag,
        str(device),
    )
    if key in OWN_AGENT_CACHE:
        return OWN_AGENT_CACHE[key]

    filename = (
        "vpr_reranking_gru_"
        + agent_tag(
            arch=arch,
            derivatives=derivatives,
            progress=progress,
            use_core=core,
            use_scale=scale,
            input_norm=innorm,
            **channels,
        )
        + th_tag
        + f"_{scene}{suffix}{seed_tag}.pth"
    )
    model_root = MODEL_IDS[trained_backbone]
    path = format_path(
        PAPER_CONFIG,
        "weight",
        model_id=model_root,
        filename=filename,
    )
    if not os.path.exists(path):
        if seed is not None:
            return None
        raise FileNotFoundError(
            f"missing cached agent for {scene}: {path}; run the matching Part B training step"
        )
    input_dim = agent_input_dim(
        derivatives=derivatives,
        progress=progress,
        use_core=core,
        use_scale=scale,
        **channels,
    )
    agent = VPRReranking(
        input_dim=input_dim,
        hidden_dim=DL_HIDDEN_DIM,
        arch=arch,
        input_norm=innorm,
    ).to(device)
    agent.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    agent.eval()
    OWN_AGENT_CACHE[key] = agent
    return agent


def _grouped_records(data_path, max_steps=MAX_STEPS):

    hit = _RECORDS_CACHE.get(data_path)
    if hit is not None:
        _RECORDS_CACHE.move_to_end(data_path)
        return hit
    with open(data_path, "r", encoding="utf-8") as handle:
        records = json.load(handle)
    from train_agent import _require_temporal_rows

    _require_temporal_rows(records, data_path, max_steps)
    grouped = defaultdict(list)
    for record in records:
        grouped[record["query"]].append(record)
    _RECORDS_CACHE[data_path] = grouped
    while len(_RECORDS_CACHE) > _RECORDS_CACHE_SLOTS:
        _RECORDS_CACHE.popitem(last=False)
    return grouped


def _run_own_partition(
    scene,
    backbone,
    variant,
    log_path,
    seed_average=False,
    threshold=None,
    agent_backbone=AGENT_BACKBONE,
    weight_scene=None,
    weight_threshold=None,
):
    threshold = OWN_TEST_THRESHOLDS[scene] if threshold is None else float(threshold)
    default_head = PAPER_CONFIG.get("default_head") or {}
    if weight_scene is None and default_head.get("scene"):
        weight_scene = str(default_head["scene"])
    if weight_threshold is None and default_head.get("threshold") is not None:
        weight_threshold = float(default_head["threshold"])
    key = (
        scene,
        backbone,
        agent_backbone,
        tuple(sorted(variant.items())),
        seed_average,
        threshold,
        weight_scene,
        weight_threshold,
    )
    if key in OWN_METRIC_CACHE:
        return OWN_METRIC_CACHE[key]

    derivatives = variant["ABLATION_USE_DERIVATIVES"] == "True"
    progress = variant["ABLATION_USE_PROGRESS"] == "True"
    core = variant["ABLATION_USE_CORE"] == "True"
    scale = variant["ABLATION_USE_SCALE"] == "True"
    channels = _variant_channels(variant)
    arch = variant["ABLATION_ARCH"]
    scene_steps = _scene_max_steps(scene)
    steps = head_steps(arch, scene_steps)
    suffix = data_tag(
        random_window=variant["DATA_ABLATION_RANDOM_WINDOW"] == "True",
    )

    data_path = format_path(
        PAPER_CONFIG,
        "trajectory",
        model_id=MODEL_IDS[backbone],
        scene=scene,
        suffix=suffix,
    )
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"missing cached test trajectories: {data_path}; run the matching Part B build step"
        )
    grouped = _grouped_records(data_path, scene_steps)
    total = len(grouped)

    raw_hit = sgv_hit = smooth_hit = fusion_hit = recallk_hit = approach_hit = 0
    winmax_hit = winsum_hit = 0
    pa_conf = []
    sue_conf = {slope: [] for slope in SUE_SLOPES}
    inlier_conf = []
    raw_oks = []
    sgv_oks = []
    feature_key = (
        data_path,
        tuple(sorted(channels.items())),
        derivatives,
        progress,
        core,
        scale,
        steps,
    )
    cached_features = _FEATURES_CACHE.get(feature_key)
    if cached_features is not None:
        _FEATURES_CACHE.move_to_end(feature_key)
    feature_groups, masks, candidate_groups = (
        cached_features if cached_features is not None else ([], [], [])
    )
    build_features = cached_features is None
    dummy = build_sequence_features(
        [],
        [],
        steps,
        use_derivatives=derivatives,
        use_progress=progress,
        use_core=core,
        use_scale=scale,
    )

    for candidates in grouped.values():
        candidates = sorted(candidates, key=lambda item: item.get("rank", 999))[:TOP_K]
        if not candidates:
            continue
        if build_features:
            candidate_groups.append(candidates)

        valid_errors = [
            item.get("error", -1.0)
            for item in candidates
            if item.get("error", -1.0) >= 0
        ]
        raw_ok = bool(valid_errors and valid_errors[0] <= threshold)
        raw_hit += raw_ok
        recallk_hit += bool(any(0 <= e <= threshold for e in valid_errors))

        best_sgv = max(
            candidates,
            key=lambda item: (
                item["inliers"][len(item["inliers"]) // 2] if item.get("inliers") else 0
            ),
        )
        sgv_ok = 0 <= best_sgv.get("error", -1.0) <= threshold
        sgv_hit += sgv_ok

        distances = [
            max(
                0.0,
                1.0 - abs(item.get("scores", [0.0])[0] if "scores" in item else 0.0),
            )
            for item in candidates
        ]
        pa_conf.append(
            -distances[0] / (distances[1] + 1e-6) if len(distances) >= 2 else -1.0
        )

        def _center_sim(item):
            scores = item.get("scores")
            return abs(scores[len(scores) // 2]) if scores else 0.0

        sue_candidates = [item for item in candidates if "global_pose" in item]
        if len(sue_candidates) >= 2:
            count = min(10, len(sue_candidates))
            poses = np.array(
                [item["global_pose"][:2] for item in sue_candidates[:count]]
            )
            d_l2 = np.array(
                [
                    math.sqrt(max(0.0, 2.0 - 2.0 * _center_sim(item)))
                    for item in sue_candidates[:count]
                ]
            )
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
                    w[k] * min(500.0, abs(poses[k, 0] - mean_pose[0])) ** 2
                    for k in range(count)
                )
                var_lon = sum(
                    w[k] * min(500.0, abs(poses[k, 1] - mean_pose[1])) ** 2
                    for k in range(count)
                )
                sue_conf[slope].append(-(float(var_lat) + float(var_lon)) / 2.0)
        else:
            for slope in SUE_SLOPES:
                sue_conf[slope].append(-float("inf"))

        inlier_conf.append(
            float(
                max(candidates[0].get("inliers", [0]))
                if candidates[0].get("inliers")
                else 0.0
            )
        )
        raw_oks.append(raw_ok)
        sgv_oks.append(sgv_ok)

        def _valid_mean_score(item):
            scores = item.get("scores")
            if not scores:
                return 0.0
            valid = item.get("step_valid") or [1.0] * len(scores)
            kept = [x for x, v in zip(scores, valid) if v > 0.5]
            if not kept:
                kept = [scores[len(scores) // 2]]
            return sum(kept) / len(kept)

        best_smooth = max(candidates, key=_valid_mean_score)
        smooth_hit += 0 <= best_smooth.get("error", -1.0) <= threshold

        def fusion_score(item):
            scores = item.get("scores", [])
            inliers = item.get("inliers", [])
            if LOG1P:
                return sum(
                    score * min(1.0, math.log1p(inlier) / 4.0)
                    for score, inlier in zip(scores, inliers)
                )
            return sum(
                score * min(1.0, inlier / INLIER_NORM_FACTOR)
                for score, inlier in zip(scores, inliers)
            )

        best_fusion = max(candidates, key=fusion_score)
        fusion_hit += 0 <= best_fusion.get("error", -1.0) <= threshold

        def approach_score(item):
            values = (
                np.log1p(np.asarray(item.get("inliers", []) or [], dtype=float)) / 4.0
            )
            if len(values) < 2 or values.std() < 1e-9:
                return -1.0
            centre = (len(values) - 1) / 2.0
            template = 1.0 - np.abs(np.arange(len(values)) - centre) / centre
            return float(np.corrcoef(values, template)[0, 1])

        best_approach = max(candidates, key=approach_score)
        approach_hit += 0 <= best_approach.get("error", -1.0) <= threshold

        def _valid_inliers(item):
            inliers = item.get("inliers") or []
            valid = item.get("step_valid") or [1.0] * len(inliers)
            kept = [value for value, flag in zip(inliers, valid) if flag > 0.5]
            return kept or [0.0]

        best_winmax = max(candidates, key=lambda item: max(_valid_inliers(item)))
        winmax_hit += 0 <= best_winmax.get("error", -1.0) <= threshold
        best_winsum = max(candidates, key=lambda item: sum(_valid_inliers(item)))
        winsum_hit += 0 <= best_winsum.get("error", -1.0) <= threshold

        if build_features:
            head_cands = (
                center_frame_view(candidates, scene_steps)
                if steps != scene_steps
                else candidates
            )
            extra = build_extra_channels(head_cands, steps, **channels)
            n_extra = len(extra[0]) if extra else 0
            features = [
                build_sequence_features(
                    item.get("inliers", []),
                    item.get("scale_ratio", []),
                    steps,
                    use_derivatives=derivatives,
                    use_progress=progress,
                    extra_channels=extra[ci] if extra else None,
                    use_core=core,
                    use_scale=scale,
                )
                for ci, item in enumerate(head_cands)
            ]
            pad = [row + [0.0] * n_extra for row in dummy]
            features.extend([pad] * (TOP_K - len(features)))
            feature_groups.append(features)
            masks.append([1.0] * len(candidates) + [0.0] * (TOP_K - len(candidates)))

    if build_features:
        _FEATURES_CACHE[feature_key] = (feature_groups, masks, candidate_groups)
        while len(_FEATURES_CACHE) > _FEATURES_CACHE_SLOTS:
            _FEATURES_CACHE.popitem(last=False)

    device = None
    if FORCE_CPU:
        torch.set_num_threads(1)
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        _record_missing(
            f"{scene}/{backbone}/SPARE",
            "CUDA is unavailable (pass --cpu for a diagnostic, non-canonical replay)",
            log_path,
        )
    gate_margin = float(os.getenv("STAR_GATE_MARGIN", "0"))
    seed_list = [None, *PAPER_SEEDS[1:]] if seed_average else [None]
    seed_hits = []
    _stacked = None
    if device is not None:
        for _seed in seed_list:
            try:
                agent = _own_agent(
                    scene,
                    variant,
                    device,
                    seed=_seed,
                    threshold=threshold,
                    trained_backbone=agent_backbone,
                    weight_scene=weight_scene,
                    weight_threshold=weight_threshold,
                )
            except (FileNotFoundError, RuntimeError, ValueError) as exc:
                _record_missing(
                    f"{scene}/{backbone}/SPARE@{threshold:g}m", exc, log_path
                )
                break
            if agent is None:
                continue
            ours_hit = 0
            if _stacked is None:
                _stacked = (
                    torch.tensor(feature_groups, dtype=torch.float32, device=device),
                    torch.tensor(masks, dtype=torch.float32, device=device),
                )
            all_x, all_mask = _stacked
            with torch.no_grad():
                for local_index in range(len(feature_groups)):
                    x = all_x[local_index : local_index + 1]
                    mask = all_mask[local_index : local_index + 1]
                    probabilities = agent.predict_softmax(x, mask=mask)[0, :, -1]
                    candidates = candidate_groups[local_index]
                    pick = int(torch.argmax(probabilities).item())
                    if pick >= len(candidates):
                        pick = 0
                    margin = float(probabilities[pick] - probabilities[0])
                    selected = pick if margin >= gate_margin else 0
                    error = candidates[selected].get("error", -1.0)
                    ours_hit += 0 <= error <= threshold
            seed_hits.append(100.0 * ours_hit / total if total else 0.0)

    if seed_average and len(seed_hits) < len(seed_list):
        print(
            f"  ⚠️  {scene} {arch} deriv={derivatives} prog={progress}"
            f"{suffix}: only {len(seed_hits)}/{len(seed_list)} seeds trained "
            f"-- this cell is NOT comparable to a full-seed one",
            flush=True,
        )

    pct = lambda count: 100.0 * count / total if total else 0.0
    metrics = {
        f"Raw VPR Base: {backbone}": pct(raw_hit),
        "Recall@K (pool ceiling)": pct(recallk_hit),
        "Naive Temporal Smoothing": pct(smooth_hit),
        "Inlier-Weighted Fusion": pct(fusion_hit),
        "Single-Frame Dense GV (LightGlue)": pct(sgv_hit),
        "PA-Selective GV": pct(_oracle_switch(pa_conf, raw_oks, sgv_oks)),
        "SUE-Selective GV": pct(
            max(
                _oracle_switch(sue_conf[slope], raw_oks, sgv_oks)
                for slope in SUE_SLOPES
            )
        ),
        "Inlier-Selective GV": pct(_oracle_switch(inlier_conf, raw_oks, sgv_oks)),
        "Hand-crafted Approach Score (no learning)": pct(approach_hit),
        "Window Max Inliers (no learning)": pct(winmax_hit),
        "Window Sum Inliers (no learning)": pct(winsum_hit),
        f"Ours (arch={arch}, deriv={derivatives}, prog={progress})": (
            sum(seed_hits) / len(seed_hits) if seed_hits else None
        ),
        "__star_seeds": seed_hits,
    }
    OWN_METRIC_CACHE[key] = metrics
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(
            f"{scene}/{backbone}/{arch}/deriv={derivatives}/prog={progress}{suffix}: "
            + json.dumps(metrics, sort_keys=True)
            + "\n"
        )
    return metrics


def _own_dataset_metrics(
    dataset,
    backbone,
    variant,
    log_path,
    seed_average=False,
    threshold=None,
    agent_backbone=AGENT_BACKBONE,
):
    start = time.time()
    partitions = OWN_DATASETS[dataset]
    per_partition = [
        _run_own_partition(
            scene,
            backbone,
            variant,
            log_path,
            seed_average=seed_average,
            threshold=threshold,
            agent_backbone=agent_backbone,
        )
        for scene in partitions
    ]
    common = set(per_partition[0])
    for metrics in per_partition[1:]:
        common.intersection_update(metrics)
    aggregate = {}
    for method in per_partition[0]:
        if method not in common or method == "__star_seeds":
            continue
        values = [metrics[method] for metrics in per_partition]
        aggregate[method] = (
            sum(values) / len(values) if all(v is not None for v in values) else None
        )
    star_key = next((k for k in aggregate if k.startswith("Ours (")), None)
    if star_key is not None:
        seed_lists = [m["__star_seeds"] for m in per_partition]
        n_seeds = min((len(sl) for sl in seed_lists), default=0)
        macro_per_seed = [
            sum(sl[i] for sl in seed_lists) / len(seed_lists) for i in range(n_seeds)
        ]
        aggregate[star_key] = (
            sum(macro_per_seed) / len(macro_per_seed) if macro_per_seed else None
        )
        if n_seeds > 1:
            aggregate["__star_std"] = statistics.stdev(macro_per_seed)
        aggregate["__star_seeds"] = macro_per_seed
    try:
        with open(log_path, "a", encoding="utf-8") as _fh:
            _fh.write(
                f"{backbone:12s} @ {dataset:12s} assets replay "
                f"{time.time() - start:5.1f}s\n"
            )
    except OSError:
        pass
    _progress_tick(f"{backbone} @ {dataset}")
    return aggregate


def _oracle_switch(confs, raw_oks, sgv_oks):

    return max(
        sum(
            raw if conf >= threshold else sgv
            for conf, raw, sgv in zip(confs, raw_oks, sgv_oks)
        )
        for threshold in sorted(set(confs))
    )


def _metrics(
    dataset,
    backbone,
    variant,
    log_path,
    seed_average=False,
    threshold=None,
    agent_backbone=AGENT_BACKBONE,
):
    return _own_dataset_metrics(
        dataset,
        backbone,
        variant,
        log_path,
        seed_average=seed_average,
        threshold=threshold,
        agent_backbone=agent_backbone,
    )


_REPLAY_MODE = "direct"
_COLLECTED = {}
_PREFETCHED = {}


def _replay_workers():
    env = os.getenv("PAPER_REPLAY_WORKERS", "auto").strip()
    if env != "auto":
        return max(1, int(env))
    cpu_cap = max(1, (os.cpu_count() or 4) - 2)
    try:
        import subprocess as _sp

        free_mb = int(
            _sp.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.free",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            ).splitlines()[0]
        )
        vram_cap = max(1, free_mb // 700)
    except Exception:
        vram_cap = 4
    try:
        with open("/proc/meminfo", encoding="ascii") as handle:
            available_gb = next(
                int(line.split()[1]) / 1048576
                for line in handle
                if line.startswith("MemAvailable:")
            )
        mem_cap = max(1, int(available_gb // 2))
    except (OSError, StopIteration):
        mem_cap = 2
    return max(1, min(cpu_cap, vram_cap, mem_cap))


def _spec_json(dataset, backbone, variant, seed_average, threshold, agent_backbone):
    return json.dumps(
        {
            "dataset": dataset,
            "backbone": backbone,
            "variant": variant,
            "seed_average": seed_average,
            "threshold": threshold,
            "agent_backbone": agent_backbone,
        },
        sort_keys=True,
    )


def _worker_init(force_cpu):
    global FORCE_CPU
    FORCE_CPU = force_cpu


def _prefetch_worker(spec):
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)

    kw = json.loads(spec)
    metrics = _safe_metrics(
        kw["dataset"],
        kw["backbone"],
        kw["variant"],
        os.devnull,
        seed_average=kw["seed_average"],
        threshold=kw["threshold"],
        agent_backbone=kw["agent_backbone"],
    )
    missing = dict(MISSING_ITEMS)
    if metrics is None:
        return spec, "__MISS__", missing
    return (
        spec,
        {
            k: (float(v) if isinstance(v, np.floating) else v)
            for k, v in metrics.items()
        },
        missing,
    )


def _safe_metrics(
    dataset,
    backbone,
    variant,
    log_path,
    seed_average=False,
    threshold=None,
    agent_backbone=AGENT_BACKBONE,
):

    if _REPLAY_MODE == "collect":
        _COLLECTED[
            _spec_json(
                dataset, backbone, variant, seed_average, threshold, agent_backbone
            )
        ] = None
        return None
    if _REPLAY_MODE == "serve":
        got = _PREFETCHED.get(
            _spec_json(
                dataset, backbone, variant, seed_average, threshold, agent_backbone
            )
        )
        if got is not None and got != "__MISS__":
            return got
    context = (
        f"{dataset}/{backbone}/"
        + agent_tag(
            arch=variant["ABLATION_ARCH"],
            derivatives=variant["ABLATION_USE_DERIVATIVES"] == "True",
            progress=variant["ABLATION_USE_PROGRESS"] == "True",
            use_core=variant["ABLATION_USE_CORE"] == "True",
            **_variant_channels(variant),
        )
        + (f"@{threshold:g}m" if threshold is not None else "")
    )
    try:
        return _metrics(
            dataset,
            backbone,
            variant,
            log_path,
            seed_average=seed_average,
            threshold=threshold,
            agent_backbone=agent_backbone,
        )
    except Exception as exc:
        _record_missing(context, exc, log_path)
        return None


def _metric_by_prefix(metrics, prefix):
    if not metrics:
        return None
    matches = [value for name, value in metrics.items() if name.startswith(prefix)]
    if len(matches) != 1:
        return None
    return matches[0]


REPORT_THRESHOLDS = tuple(float(v) for v in PAPER_CONFIG["thresholds_m"])
SWEEP_THRESHOLDS = (None, *REPORT_THRESHOLDS[1:])


def _fmt1(value):

    if value is None or not isinstance(value, (int, float, np.number)):
        return "--"
    return f"{value:.1f}"


def _format(dataset, value):
    if value is None or not isinstance(value, (int, float, np.number)):
        return MISSING_CELL
    decimals = int(PAPER_CONFIG["datasets"][dataset]["decimals"])
    return f"{value:.{decimals}f}"


def _format_star(dataset, metrics):

    raw_value = _metric_by_prefix(metrics, "Ours (")
    if raw_value is None:
        return MISSING_CELL
    value = _format(dataset, raw_value)
    std = metrics.get("__star_std")
    return f"{value}±{std:.1f}" if std is not None else value


_PROGRESS = {"n": 0, "dirty": False}


def _progress_tick(label, done=None, total=None):
    _PROGRESS["n"] += 1
    n = done if done is not None else _PROGRESS["n"]
    suffix = f"{n}/{total}" if total else f"{n}"
    if sys.stdout.isatty():
        sys.stdout.write(f"\r  [replay] {suffix} cells | {label:<32.32s}")
        sys.stdout.flush()
        _PROGRESS["dirty"] = True
    elif n % 10 == 0 or (total and n == total):
        print(f"  [replay] {suffix} cells done ({label})", flush=True)


def _progress_clear():
    if _PROGRESS["dirty"]:
        sys.stdout.write("\r" + " " * 60 + "\r")
        sys.stdout.flush()
        _PROGRESS["dirty"] = False


_RENDERED = {}


def _print_table(title, headers, rows, log_path):
    _progress_clear()
    if _REPLAY_MODE == "collect":
        return
    _RENDERED[title] = (list(headers), [list(r) for r in rows])
    lines = [f"\n## {title}", "| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join(["---"] + ["---:" for _ in headers[1:]]) + "|")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    output = "\n".join(lines) + "\n"
    print(output)
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(output)


def _baseline_values(method, datasets, field="value"):

    baseline = PAPER_CONFIG["classic_baselines"][method]
    slug = baseline["slug"]
    values = {}
    for dataset in datasets:
        scenes = OWN_DATASETS[dataset]
        radii = _dataset_radii(dataset)
        per_th = {th: [] for th in radii}
        failed = False
        for th in radii:
            for scene in scenes:
                path = os.path.join(
                    PAPER_CONFIG["paths"]["classic_baselines"],
                    f"{slug}_{scene}_no_roi.json",
                )
                try:
                    with open(path, "r", encoding="utf-8") as handle:
                        result = json.load(handle)
                    if (
                        result.get("method") != method
                        or int(result.get("queries", 0)) <= 0
                    ):
                        raise ValueError("wrong method or zero-query result")
                    multi = "raw_values" if field == "raw_value" else "values"
                    th_key = str(float(th))
                    if multi in result and th_key in result[multi]:
                        per_th[th].append(float(result[multi][th_key]))
                    elif abs(float(result.get("threshold_m", -1.0)) - th) < 1e-9:
                        per_th[th].append(float(result[field]))
                    else:
                        sfx = os.path.join(
                            PAPER_CONFIG["paths"]["classic_baselines"],
                            f"{slug}_{scene}_th{th:g}_no_roi.json",
                        )
                        if not os.path.isfile(sfx):
                            continue
                        with open(sfx, "r", encoding="utf-8") as handle:
                            r2 = json.load(handle)
                        if r2.get("method") != method or int(r2.get("queries", 0)) <= 0:
                            raise ValueError("wrong method or zero-query result")
                        per_th[th].append(float(r2[field]))
                except (OSError, ValueError, TypeError, KeyError) as exc:
                    _record_missing(f"{method}/{scene}@{th:g}m", f"{path}: {exc}")
                    failed = True
                    break
            if failed:
                break
        if not failed:
            values[dataset] = [
                (sum(vs) / len(vs)) if vs else None for vs in per_th.values()
            ]
    return values


def _valid_reranker_result(result, recorded_method, scene, radii=None):

    radii = REPORT_THRESHOLDS if radii is None else radii
    metadata = result.get("metadata") or {}
    try:
        count = int(result.get("n_queries", 0))
        recalls = result["reranked_recall"]
        return (
            metadata.get("method") == recorded_method
            and metadata.get("scene") == scene
            and count > 0
            and all(f"{threshold:g}" in recalls for threshold in radii)
        )
    except (KeyError, TypeError, ValueError):
        return False


def _learned_reranker_values(method, datasets, pool=None):

    baseline = (
        PAPER_CONFIG["learned_baselines"].get(method)
        or PAPER_CONFIG.get("report_only_baselines", {})[method]
    )
    slug = baseline["slug"]
    recorded_method = baseline["recorded_method"]
    per_threshold = bool(baseline.get("per_threshold"))
    pool = pool or MODEL_IDS[MAIN_BACKBONE]

    def _paths_for(scene, threshold):

        if per_threshold:
            sl = baseline["slug_pattern"].format(th=threshold)
            rm = baseline["recorded_method_pattern"].format(th=threshold)
        else:
            sl, rm = slug, recorded_method
        return (
            os.path.join(
                PAPER_CONFIG["paths"]["learned_baselines"],
                f"{sl}_{scene}__pool-{pool}.json",
            ),
            rm,
        )

    values = {}
    for dataset in datasets:
        radii = _dataset_radii(dataset)
        route_values = {threshold: [] for threshold in radii}
        failed = False
        for scene in OWN_DATASETS[dataset]:
            try:
                for threshold in radii:
                    path, rm = _paths_for(scene, threshold)
                    with open(path, encoding="utf-8") as handle:
                        result = json.load(handle)
                    if not _valid_reranker_result(result, rm, scene, radii):
                        raise ValueError(
                            "wrong evaluator, method, scene, or query count"
                        )
                    recalls = result["reranked_recall"]
                    route_values[threshold].append(float(recalls[f"{threshold:g}"]))
                    if not per_threshold:
                        for other in radii:
                            if other != threshold:
                                route_values[other].append(float(recalls[f"{other:g}"]))
                        break
            except (OSError, ValueError, TypeError, KeyError) as exc:
                path, _ = _paths_for(scene, radii[0])
                _record_missing(f"{method}/{scene}", f"{path}: {exc}")
                failed = True
                break
        if not failed:
            values[dataset] = [
                sum(route_values[threshold]) / len(route_values[threshold])
                for threshold in radii
            ]
    return values


def _dataset_radii(dataset):

    scenes = OWN_DATASETS.get(dataset) or [dataset]
    own = float(OWN_TEST_THRESHOLDS.get(scenes[0], REPORT_THRESHOLDS[0]))
    return tuple(sorted((own, *REPORT_THRESHOLDS[1:])))


def _sweep_order(dataset):

    scenes = OWN_DATASETS.get(dataset) or [dataset]
    own = float(OWN_TEST_THRESHOLDS.get(scenes[0], REPORT_THRESHOLDS[0]))
    return tuple(
        sorted(SWEEP_THRESHOLDS, key=lambda th: own if th is None else float(th))
    )


def _table_main(datasets, log_path):
    mega = {
        (dataset, th): _safe_metrics(
            dataset,
            MAIN_BACKBONE,
            FULL_VARIANT,
            log_path,
            seed_average=True,
            threshold=th,
        )
        for dataset in datasets
        if dataset not in PLACEHOLDER_DATASETS
        for th in SWEEP_THRESHOLDS
    }

    def cell(dataset, prefix):
        if dataset in PLACEHOLDER_DATASETS:
            return MISSING_CELL
        return "/".join(
            _fmt1(_metric_by_prefix(mega[(dataset, th)], prefix))
            for th in _sweep_order(dataset)
        )

    def cell_aux(dataset, values):
        if dataset in PLACEHOLDER_DATASETS or dataset not in values:
            return MISSING_CELL
        return "/".join(_fmt1(v) if v is not None else "--" for v in values[dataset])

    def cell_star(dataset):
        if dataset in PLACEHOLDER_DATASETS:
            return MISSING_CELL
        parts = []
        for th in _sweep_order(dataset):
            metrics = mega[(dataset, th)]
            value = _metric_by_prefix(metrics, "Ours (")
            text = _fmt1(value)
            std = (metrics or {}).get("__star_std")
            if th is None and value is not None and std is not None:
                text += f"±{std:.1f}"
            parts.append(text)
        return "/".join(parts)

    rows = []
    classic_cache = {}
    learned_cache = {}
    for spec in PAPER_CONFIG["main_rows"]:
        kind = spec["kind"]
        label = spec["label"]
        if kind == "metric":
            values = [cell(dataset, spec["prefix"]) for dataset in datasets]
        elif kind == "classic":
            cache_key = (spec["method"], spec.get("field", "value"))
            if cache_key not in classic_cache:
                classic_cache[cache_key] = _baseline_values(
                    spec["method"],
                    datasets,
                    field=cache_key[1],
                )
            values = [
                cell_aux(dataset, classic_cache[cache_key]) for dataset in datasets
            ]
        elif kind == "learned":
            method = spec["method"]
            pool = spec.get("pool") or MODEL_IDS[MAIN_BACKBONE]
            cache_key = (method, pool)
            if cache_key not in learned_cache:
                learned_cache[cache_key] = _learned_reranker_values(
                    method, datasets, pool=pool
                )
            learned_cache[method] = learned_cache[cache_key]
            values = [
                "/".join(
                    _fmt1(value) for value in learned_cache[method].get(dataset, [])
                )
                or MISSING_CELL
                for dataset in datasets
            ]
        elif kind == "star":
            values = [cell_star(dataset) for dataset in datasets]
        else:
            raise ValueError(f"unknown main-table row kind: {kind!r}")
        rows.append([label, *values])
    _print_table(PAPER_CONFIG["tables"]["main"], ["Method"] + datasets, rows, log_path)


def _table_backbones(datasets, backbones, log_path, retrained=False):
    headers = ["Backbone", ""] + list(datasets)
    rows = []
    for backbone in backbones:
        raw_row = [backbone, "Raw"]
        star_row = ["", "+SPARE"]
        for dataset in datasets:
            if dataset in PLACEHOLDER_DATASETS:
                raw_row.append(MISSING_CELL)
                star_row.append(MISSING_CELL)
                continue
            raw_parts, star_parts = [], []
            for th in _sweep_order(dataset):
                metrics = _safe_metrics(
                    dataset,
                    backbone,
                    FULL_VARIANT,
                    log_path,
                    seed_average=True,
                    threshold=th,
                    agent_backbone=backbone if retrained else AGENT_BACKBONE,
                )
                raw_parts.append(_fmt1(_metric_by_prefix(metrics, "Raw VPR Base:")))
                star_parts.append(_fmt1(_metric_by_prefix(metrics, "Ours (")))
            raw_row.append("/".join(raw_parts))
            star_row.append("/".join(star_parts))
        rows.append(raw_row)
        rows.append(star_row)
    title_key = "backbones_retrained" if retrained else "backbones"
    title = PAPER_CONFIG["tables"][title_key]
    _print_table(title, headers, rows, log_path)


ABLATION_DATASETS = [
    name for name, spec in PAPER_CONFIG["datasets"].items() if spec.get("ablation")
]


def _ablation_cell(dataset, variant, log_path, prefix="Ours (", seed_average=True):

    parts = []
    for th in _sweep_order(dataset):
        metrics = _safe_metrics(
            dataset,
            MAIN_BACKBONE,
            variant,
            log_path,
            seed_average=seed_average,
            threshold=th,
        )
        value = _metric_by_prefix(metrics, prefix)
        if value is None:
            parts.append("--")
            continue
        std = metrics.get("__star_std") if seed_average else None
        parts.append(_fmt1(value) + (f"±{std:.1f}" if std is not None else ""))
    return "/".join(parts)


def _table_ablation(datasets, log_path):
    own_datasets = [
        dataset
        for dataset in datasets
        if dataset in OWN_DATASETS and dataset in ABLATION_DATASETS
    ]
    rows = []
    raw_row = ["Raw VPR (MegaLoc, no re-ranking)"]
    for dataset in own_datasets:
        raw_row.append(
            _ablation_cell(
                dataset,
                FULL_VARIANT,
                log_path,
                prefix="Raw VPR Base:",
                seed_average=False,
            )
        )
    rows.append(raw_row)
    for label, variant in ABLATION_VARIANTS:
        row = [label]
        for dataset in own_datasets:
            if variant is None:
                row.append(
                    _ablation_cell(
                        dataset,
                        FULL_VARIANT,
                        log_path,
                        prefix=label,
                        seed_average=False,
                    )
                )
            else:
                row.append(_ablation_cell(dataset, variant, log_path))
        rows.append(row)
    _print_table(
        PAPER_CONFIG["tables"]["ablation"],
        ["Variant"] + own_datasets,
        rows,
        log_path,
    )


_NA_TOKENS = {MISSING_CELL, "--", ""}
_NA_OUT = "--"


def _cell3(cell):

    return "/".join([_NA_OUT] * 3) if cell in _NA_TOKENS else cell


def _is_missing(value):

    text = str(value).strip()
    na = _NA_TOKENS | {_NA_OUT, ""}
    if text in na:
        return True
    parts = [part.strip() for part in text.split("/")]
    return len(parts) > 1 and all(part in na for part in parts)


def _half_up_mean(tokens):

    from decimal import Decimal, ROUND_HALF_UP, InvalidOperation

    if any(t in _NA_TOKENS for t in tokens):
        return _NA_OUT
    try:
        total = sum(Decimal(t) for t in tokens)
    except InvalidOperation:
        raise ValueError(
            f"Mean cannot parse token: {tokens!r} -- a rendered cell contains "
            f"a non-numeric value other than the missing marker; fix upstream before emitting"
        )
    return str((total / len(tokens)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def _mean_cell(cells):
    parts = [c.split("/") for c in cells]
    return "/".join(_half_up_mean([p[k] for p in parts]) for k in range(3))


def _errors_recall(path, key, radius):

    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    errors = payload[key]
    hits = sum(1 for e in errors if e is not None and e <= radius)
    return hits / len(errors) * 100.0


def _baseline_json(slug, scene, pool):
    return os.path.join(
        "assets", "baseline_results", f"{slug}_{scene}__pool-{pool}.json"
    )


def _table_threshold25(datasets, log_path):

    if _REPLAY_MODE == "collect":
        return
    scenes = ["RobotCar", "4Seasons", "NCLT", "Boreas"]
    if not set(scenes) <= set(datasets):
        print(
            "[threshold25] skipped: needs RobotCar, 4Seasons, NCLT and Boreas in --datasets"
        )
        return
    pool = MODEL_IDS[MAIN_BACKBONE]
    per = {}
    for scene in scenes:
        per[scene] = _own_dataset_metrics(
            scene,
            MAIN_BACKBONE,
            FULL_VARIANT,
            log_path,
            seed_average=True,
            threshold=25.0,
        )
    rows = []

    def metric_row(label, key, n_seeds=1, star=False):
        cells = []
        for scene in scenes:
            m = per[scene]
            v = m.get(key) if not star else _metric_by_prefix(m, "Ours (")
            if v is None:
                _record_missing(
                    f"{scene}/threshold25/{label}", "metric missing", log_path
                )
                cells.append(MISSING_CELL)
                continue
            text = f"{v:.1f}"
            if star:
                seeds_v = m.get("__star_seeds") or []
                if len(seeds_v) > 1:
                    mu = sum(seeds_v) / len(seeds_v)
                    pstd = (sum((x - mu) ** 2 for x in seeds_v) / len(seeds_v)) ** 0.5
                    text += f"+-{pstd:.1f}"
            cells.append(text)
        rows.append([label, *cells, str(n_seeds), "25"])

    metric_row("Raw MegaLoc", "Raw VPR Base: MegaLoc")
    metric_row("Single-frame dense GV (LightGlue)", "Single-Frame Dense GV (LightGlue)")
    for label, slug in (("FoL", "fol_rerank"), ("Pair-VPR (stage 2)", "pairvpr_s2")):
        cells = []
        for scene in scenes:
            parts = OWN_DATASETS.get(scene, [scene])
            try:
                vals = [
                    _errors_recall(
                        _baseline_json(slug, part, pool), "reranked_errors_m", 25.0
                    )
                    for part in parts
                ]
                cells.append(f"{sum(vals) / len(vals):.1f}")
            except (OSError, KeyError) as exc:
                _record_missing(f"{scene}/threshold25/{label}", exc, log_path)
                cells.append(MISSING_CELL)
        rows.append([label, *cells, "1", "25"])
    metric_row("SPARE (ours)", None, n_seeds=6, star=True)
    metric_row("Pool upper bound (Top-10)", "Recall@K (pool ceiling)")
    order = [
        "Raw MegaLoc",
        "Single-frame dense GV (LightGlue)",
        "FoL",
        "Pair-VPR (stage 2)",
        "SPARE (ours)",
        "Pool upper bound (Top-10)",
    ]
    rows.sort(key=lambda r: order.index(r[0]))
    _print_table(
        "Success-radius sensitivity (25 m)",
        ["Method", *scenes],
        [r[: 1 + len(scenes)] for r in rows],
        log_path,
    )
    if _REPLAY_MODE != "collect":
        csv_dir = PAPER_CONFIG["paper_csv"]["dir"]
        out = os.path.join(csv_dir, "threshold25.csv")
        os.makedirs(csv_dir, exist_ok=True)
        with open(out, "w", encoding="utf-8", newline="\n") as fh:
            fh.write("method," + ",".join(scenes) + ",n_seeds,radius_m,source\n")
            for r in rows:
                fh.write(",".join(r) + ",run_paper_tables.py\n")
        print(f"[paper_csv] threshold25 csv -> {csv_dir}")


def _table_crop_sweep(datasets, log_path):

    if _REPLAY_MODE == "collect":
        return
    if "RobotCar" not in datasets:
        print("[crop_sweep] skipped: needs RobotCar in --datasets")
        return
    pool_backbone = "BoQ"
    pool = MODEL_IDS[pool_backbone]
    columns = [
        ("RobotCarCa3040", "0.3-0.4 (paper)"),
        ("RobotCarCa4050", "0.4-0.5"),
        ("RobotCarCa5060", "0.5-0.6"),
        ("RobotCarCa7080", "0.7-0.8"),
        ("RobotCarCfull", "1.0 (no crop)"),
    ]
    labels = [label for _, label in columns]
    rows = {name: [] for name in ("Raw BoQ", "FoL", "PairVPR-s2", "VOP", "SPARE (mix)")}
    for scene, _label in columns:
        for name, slug, key in (
            ("Raw BoQ", "fol_rerank", "raw_errors_m"),
            ("FoL", "fol_rerank", "reranked_errors_m"),
            ("PairVPR-s2", "pairvpr_s2", "reranked_errors_m"),
            ("VOP", "vop_rerank", "reranked_errors_m"),
        ):
            path = _baseline_json(slug, scene, pool)
            try:
                rows[name].append(f"{_errors_recall(path, key, 5.0):.1f}")
            except (OSError, KeyError) as exc:
                _record_missing(f"{scene}/crop_sweep/{name}", exc, log_path)
                rows[name].append(MISSING_CELL)
        try:
            m = _run_own_partition(
                scene,
                pool_backbone,
                FULL_VARIANT,
                log_path,
                seed_average=True,
                threshold=5.0,
                agent_backbone=pool_backbone,
                weight_scene="RobotCarCmix",
            )
            star = _metric_by_prefix(m, "Ours (")
            seeds = m.get("__star_seeds") or []
            if star is None:
                raise RuntimeError("SPARE (mix) metric missing")
            cell = f"{star:.1f}"
            if len(seeds) > 1:
                mean = sum(seeds) / len(seeds)
                std = (sum((v - mean) ** 2 for v in seeds) / (len(seeds) - 1)) ** 0.5
                cell += f"\u00b1{std:.1f}"
            rows["SPARE (mix)"].append(cell)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            _record_missing(f"{scene}/crop_sweep/SPARE (mix)", exc, log_path)
            rows["SPARE (mix)"].append(MISSING_CELL)
    table_rows = [[name, *rows[name]] for name in rows]
    _print_table(
        "Query field-of-view robustness (RobotCar, BoQ pool)",
        ["Method", *labels],
        table_rows,
        log_path,
    )
    csv_dir = PAPER_CONFIG["paper_csv"]["dir"]
    out = os.path.join(csv_dir, f"crop_sweep_{pool}.csv")
    os.makedirs(csv_dir, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("method," + ",".join(labels) + "\n")
        for name in rows:
            fh.write(name + "," + ",".join(rows[name]) + "\n")
    print(f"[paper_csv] crop_sweep csv -> {csv_dir}")


def _mean_std_cell(seed_lists):

    lists = [lst for lst in seed_lists if lst]
    if not lists:
        return None
    n = min(len(lst) for lst in lists)
    per_seed = [sum(lst[i] for lst in lists) / len(lists) for i in range(n)]
    mean = sum(per_seed) / n
    if n > 1:
        std = (sum((v - mean) ** 2 for v in per_seed) / (n - 1)) ** 0.5
        return f"{mean:.1f}\u00b1{std:.1f}"
    return f"{mean:.1f}"


def _trajectory_datasets(datasets):

    out = []
    for dataset in datasets:
        if dataset in PLACEHOLDER_DATASETS:
            continue
        if all(abs(OWN_TEST_THRESHOLDS[s] - 5.0) < 1e-9 for s in OWN_DATASETS[dataset]):
            out.append(dataset)
    return out


def _flat_datasets(datasets):

    traj = set(_trajectory_datasets(datasets))
    return [d for d in datasets if d not in PLACEHOLDER_DATASETS and d not in traj]


def _write_csv(name, header, rows):
    csv_dir = PAPER_CONFIG["paper_csv"]["dir"]
    os.makedirs(csv_dir, exist_ok=True)
    out = os.path.join(csv_dir, name)
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(",".join(header) + "\n")
        for row in rows:
            fh.write(",".join(row) + "\n")
    print(f"[paper_csv] {name} -> {csv_dir}")


def _table_crossmap(datasets, log_path):

    if _REPLAY_MODE == "collect":
        return
    ds_list = _trajectory_datasets(datasets)
    if len(ds_list) < 2:
        print("[crossmap] skipped: needs at least two trajectory datasets")
        return

    def cell(target, weight_scenes, threshold=None, weight_threshold=None):
        seed_lists = []
        for target_scene in OWN_DATASETS[target]:
            for ws in weight_scenes(target_scene):
                m = _run_own_partition(
                    target_scene,
                    MAIN_BACKBONE,
                    FULL_VARIANT,
                    log_path,
                    seed_average=True,
                    threshold=threshold,
                    weight_scene=ws,
                    weight_threshold=weight_threshold,
                )
                seeds = m.get("__star_seeds") or []
                if not seeds:
                    raise RuntimeError(
                        f"no SPARE seeds for {target_scene} with head {ws}"
                    )
                seed_lists.append(seeds)
        return _mean_std_cell(seed_lists) or MISSING_CELL

    def safe(tag, fn):
        try:
            return fn()
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            _record_missing(tag, exc, log_path)
            return MISSING_CELL

    table_rows = []
    for source in ds_list:
        row = [source]
        for target in ds_list:
            ws = (
                (lambda sc: [sc])
                if source == target
                else (lambda sc, s=source: list(OWN_DATASETS[s]))
            )
            row.append(
                safe(f"crossmap/{source}->{target}", lambda t=target, w=ws: cell(t, w))
            )
        table_rows.append(row)
    for label, weight_of in (
        ("All maps (UnionMaps)", lambda target: "UnionMaps"),
        ("Leave-one-map-out", lambda target: f"LOO_{target}"),
    ):
        row = [label]
        for target in ds_list:
            w = weight_of(target)
            row.append(
                safe(
                    f"crossmap/{label}->{target}",
                    lambda t=target, w=w: cell(t, lambda sc: [w]),
                )
            )
        table_rows.append(row)
    _print_table(
        "Cross-map transfer (rows: head trained on; columns: judged on; 5 m; diagonal = per-map head)",
        ["Train \\ Test", *ds_list],
        table_rows,
        log_path,
    )
    _write_csv("cross_map.csv", ["train_on", *ds_list], table_rows)

    flat_list = _flat_datasets(datasets)
    if not flat_list:
        return
    flat_radii = (10.0, 15.0, 25.0)
    cols = [(d, r) for d in flat_list for r in flat_radii]
    sources = [(s, lambda sc, s=s: list(OWN_DATASETS[s])) for s in ds_list]
    sources.append(("All maps (UnionMaps)", lambda sc: ["UnionMaps"]))
    flat_rows = []
    for label, ws in sources:
        row = [label]
        for d, r in cols:
            row.append(
                safe(
                    f"crossmap_flat/{label}->{d}@{r:g}",
                    lambda d=d, r=r, ws=ws: cell(d, ws, threshold=r),
                )
            )
        flat_rows.append(row)
    row = ["Own database (self-supervised)"]
    for d, r in cols:
        row.append(
            safe(
                f"crossmap_flat/own->{d}@{r:g}",
                lambda d=d, r=r: cell(
                    d, lambda sc: [sc], threshold=r, weight_threshold=r
                ),
            )
        )
    flat_rows.append(row)
    _print_table(
        "Cross-map transfer to the benchmarks (rows: head trained on; columns: benchmark @ radius)",
        ["Train \\ Test", *[f"{d}@{r:g}" for d, r in cols]],
        flat_rows,
        log_path,
    )
    _write_csv(
        "cross_map_flat.csv", ["train_on", *[f"{d}@{r:g}" for d, r in cols]], flat_rows
    )


def _table_fixedhead(datasets, log_path):

    if _REPLAY_MODE == "collect":
        return
    radii = tuple(float(v) for v in REPORT_THRESHOLDS)
    ds_list = _trajectory_datasets(datasets)
    if not ds_list:
        print("[fixedhead] skipped: no trajectory datasets selected")
        return
    headers = ["Dataset", "Head radius"] + [f"@{r:g} m" for r in radii]
    table_rows, csv_rows = [], []
    for dataset in ds_list:
        for train_th in radii:
            row = [dataset, f"{train_th:g} m"]
            for eval_th in radii:
                seed_lists = []
                try:
                    for scene in OWN_DATASETS[dataset]:
                        m = _run_own_partition(
                            scene,
                            MAIN_BACKBONE,
                            FULL_VARIANT,
                            log_path,
                            seed_average=True,
                            threshold=eval_th,
                            weight_threshold=train_th,
                        )
                        seeds = m.get("__star_seeds") or []
                        if not seeds:
                            raise RuntimeError(
                                f"no SPARE seeds for {scene} head@{train_th:g} eval@{eval_th:g}"
                            )
                        seed_lists.append(seeds)
                    cell = _mean_std_cell(seed_lists) or MISSING_CELL
                except (FileNotFoundError, RuntimeError, ValueError) as exc:
                    _record_missing(
                        f"fixedhead/{dataset}/head{train_th:g}@{eval_th:g}",
                        exc,
                        log_path,
                    )
                    cell = MISSING_CELL
                row.append(cell)
            table_rows.append(row)
            csv_rows.append([dataset, f"{train_th:g}", *row[2:]])
    _print_table(
        "Fixed head across success radii (seed mean+/-std)",
        headers,
        table_rows,
        log_path,
    )
    _write_csv(
        "fixed_head.csv",
        ["dataset", "head_radius_m", *[f"at{r:g}m" for r in radii]],
        csv_rows,
    )

    flat_list = _flat_datasets(datasets)
    if not flat_list:
        return
    flat_radii = (10.0, 15.0, 25.0)
    headers = ["Dataset", "Head radius"] + [f"@{r:g} m" for r in flat_radii]
    table_rows, csv_rows = [], []
    for dataset in flat_list:
        for train_th in (float(REPORT_THRESHOLDS[0]), *flat_radii):
            row = [dataset, f"{train_th:g} m"]
            for eval_th in flat_radii:
                seed_lists = []
                try:
                    for scene in OWN_DATASETS[dataset]:
                        m = _run_own_partition(
                            scene,
                            MAIN_BACKBONE,
                            FULL_VARIANT,
                            log_path,
                            seed_average=True,
                            threshold=eval_th,
                            weight_threshold=train_th,
                        )
                        seeds = m.get("__star_seeds") or []
                        if not seeds:
                            raise RuntimeError(
                                f"no SPARE seeds for {scene} head@{train_th:g} eval@{eval_th:g}"
                            )
                        seed_lists.append(seeds)
                    cell = _mean_std_cell(seed_lists) or MISSING_CELL
                except (FileNotFoundError, RuntimeError, ValueError) as exc:
                    _record_missing(
                        f"fixedhead_flat/{dataset}/head{train_th:g}@{eval_th:g}",
                        exc,
                        log_path,
                    )
                    cell = MISSING_CELL
                row.append(cell)
            table_rows.append(row)
            csv_rows.append([dataset, f"{train_th:g}", *row[2:]])
    _print_table(
        "Fixed head across success radii -- flat benchmarks (seed mean+/-std)",
        headers,
        table_rows,
        log_path,
    )
    _write_csv(
        "fixed_head_flat.csv",
        ["dataset", "head_radius_m", *[f"at{r:g}m" for r in flat_radii]],
        csv_rows,
    )


def _emit_paper_csvs(log_path):

    import csv as _csv

    reg = PAPER_CONFIG["paper_csv"]
    out_dir = reg["dir"]
    os.makedirs(out_dir, exist_ok=True)
    _preserved = []

    def _write_merged(path, header, rows, key_len):

        _META = {"cite", "pool", "source", "est"}
        data_cols = [i for i in range(key_len, len(header)) if header[i] not in _META]
        for row in rows:
            for i in data_cols:
                if _is_missing(row[i]):
                    _preserved.append(f"{os.path.basename(path)}:{row[0]}:{header[i]}")
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = _csv.writer(fh)
            w.writerow(header)
            w.writerows(rows)

    ds = list(reg["datasets"])

    _rendered_ds = set()
    for _t in _RENDERED.values():
        _rendered_ds |= set(_t[0])
    _uncovered = [d for d in ds if d not in _rendered_ds]
    if _uncovered:
        print(
            f"[paper_csv] skipping write: {', '.join(_uncovered)} not rendered "
            f"this run -- partial replays never write CSVs, lest uncovered "
            f"columns be wiped to missing"
        )
        return

    main_title = PAPER_CONFIG["tables"]["main"]
    bb_title = PAPER_CONFIG["tables"]["backbones"]
    abl_title = PAPER_CONFIG["tables"]["ablation"]
    for t in (main_title, bb_title, abl_title):
        if t not in _RENDERED:
            raise RuntimeError(f"paper_csv requires {t!r} to be rendered first")

    def table_cells(title, label):
        headers, rows = _RENDERED[title]
        for row in rows:
            if row[0] == label:
                return {headers[i]: row[i] for i in range(len(headers))}
        raise KeyError(
            f"row {label!r} not found in {title!r} -- row list and renderer have drifted; refusing to emit"
        )

    def backbone_cells(backbone, kind):
        headers, rows = _RENDERED[bb_title]
        for i, row in enumerate(rows):
            if row[0] == backbone and row[1] == "Raw":
                target = row if kind == "Raw" else rows[i + 1]
                return {headers[j]: target[j] for j in range(2, len(headers))}
        raise KeyError(f"{backbone}/{kind} not found in the backbone table")

    main_rows_out = []
    for spec in reg["main"]:
        src = spec["from"]
        if src == "main_row":
            got = table_cells(main_title, spec["label"])
            cells = [_cell3(got.get(d, MISSING_CELL)) for d in ds]
        elif src == "backbone":
            got = backbone_cells(spec["backbone"], spec["kind"])
            cells = [_cell3(got.get(d, MISSING_CELL)) for d in ds]
        elif src == "learned":
            vals = _learned_reranker_values(
                spec["method_key"], ds, pool=spec["learned_pool"]
            )
            cells = []
            for d in ds:
                triple = (vals or {}).get(d)
                cells.append(
                    "/".join(_fmt1(v) for v in triple)
                    if triple
                    else _cell3(MISSING_CELL)
                )
        else:
            raise ValueError(f"unknown paper_csv source {src!r}")
        main_rows_out.append(
            [
                spec["panel"],
                spec["method"],
                spec.get("cite", "") or "",
                spec["pool"],
                *cells,
                _mean_cell(cells),
                f"replay:{src}:"
                f"{spec.get('label') or spec.get('method_key') or spec.get('backbone')}",
                "",
            ]
        )
    _write_merged(
        os.path.join(out_dir, "main.csv"),
        ["panel", "method", "cite", "pool", *ds, "Mean", "source", "est"],
        main_rows_out,
        key_len=4,
    )

    backbones = list(PAPER_CONFIG["backbones"])
    per = {(b, k): backbone_cells(b, k) for b in backbones for k in ("Raw", "+SPARE")}
    bb_rows = [
        [d, k, *[_cell3(per[(b, k)].get(d, MISSING_CELL)) for b in backbones]]
        for d in ds
        for k in ("Raw", "+SPARE")
    ]
    bb_rows += [
        [
            "Mean",
            k,
            *[
                _mean_cell([_cell3(per[(b, k)].get(d, MISSING_CELL)) for d in ds])
                for b in backbones
            ],
        ]
        for k in ("Raw", "+SPARE")
    ]
    _write_merged(
        os.path.join(out_dir, "backbones.csv"),
        ["dataset", "kind", *backbones],
        bb_rows,
        key_len=2,
    )

    abl_ds = list(ABLATION_DATASETS)
    abl_rows_out = []
    for short, full in reg["ablation_variant_labels"]:
        got = table_cells(abl_title, full)
        abl_rows_out.append(
            [
                short,
                full,
                *[
                    _NA_OUT if got.get(d, _NA_OUT) in _NA_TOKENS else got[d]
                    for d in abl_ds
                ],
            ]
        )
    _write_merged(
        os.path.join(out_dir, "ablation.csv"),
        ["variant", "variant_full", *abl_ds],
        abl_rows_out,
        key_len=2,
    )
    if _preserved:
        print(
            f"[csv] WARNING: {len(_preserved)} cell(s) could not be computed this run; written as missing (old values are never kept)"
        )

    import subprocess as _sp

    try:
        commit = _sp.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=_sp.DEVNULL
        ).strip()
    except Exception:
        commit = "unknown"
    with open(os.path.join(out_dir, "provenance.json"), "w", encoding="utf-8") as fh:
        json.dump(
            {
                "generated": datetime.now().isoformat(timespec="seconds"),
                "generator": "run_paper_tables.py::_emit_paper_csvs",
                "commit": commit,
                "datasets": ds,
                "note": "The three CSVs are generated directly by the replay; "
                "the row list lives in the paper_csv section of "
                "configs/paper_tables.yaml, and rows absent from the paper are "
                "never produced.",
            },
            fh,
            ensure_ascii=False,
            indent=2,
        )

    legacy_dir = PAPER_CONFIG["paths"]["csv"]
    for legacy in (
        "table_i_main_re_ranker_comparison.csv",
        "table_ii_cross_backbone_plug_and_play.csv",
        "table_ii_cross_backbone_retrained.csv",
        "table_iii_ablation.csv",
    ):
        path = os.path.join(legacy_dir, legacy)
        if os.path.isfile(path):
            os.remove(path)
    print(f"[paper_csv] main/backbones/ablation csv -> {out_dir}")


def main():
    global FORCE_CPU
    args = _parse_args()
    FORCE_CPU = args.cpu
    log_dir = PAPER_CONFIG["paths"]["logs"]
    os.makedirs(log_dir, exist_ok=True)
    log_path = (
        f"{log_dir}/paper_tables_replay_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )

    print("=" * 76)
    print("SPARE complete paper-table replay (assets only; no feature extraction)")
    print(f"Tables: {', '.join(args.tables)}")
    print(f"Datasets: {', '.join(args.datasets)}")
    print(f"Backbones: {', '.join(args.backbones)}")
    if FORCE_CPU:
        replay_device = "CPU (diagnostic fallback)"
    elif torch.cuda.is_available():
        replay_device = "GPU (canonical replay)"
    else:
        replay_device = "CUDA unavailable (non-agent metrics only; SPARE cells use '-')"
    print(f"Replay device: {replay_device}")
    if FORCE_CPU:
        print(
            "Warning: near-tied candidates can differ from the canonical CUDA protocol."
        )
    print(f"Detailed evaluator log: {log_path}")
    if _SCRUBBED:
        print(
            "Ignored inherited environment (would have changed these numbers): "
            + ", ".join(f"{k}={v}" for k, v in _SCRUBBED.items())
        )
    print("=" * 76)

    def _run_tables():
        if "main" in args.tables:
            _table_main(args.datasets, log_path)
        if "backbones" in args.tables:
            _table_backbones(args.datasets, args.backbones, log_path)
        if "backbones_retrained" in args.tables:
            _table_backbones(args.datasets, args.backbones, log_path, retrained=True)
        if "ablation" in args.tables:
            _table_ablation(args.datasets, log_path)
        if "threshold25" in args.tables:
            _table_threshold25(args.datasets, log_path)
        if "crop_sweep" in args.tables:
            _table_crop_sweep(args.datasets, log_path)
        if "crossmap" in args.tables:
            _table_crossmap(args.datasets, log_path)
        if "fixedhead" in args.tables:
            _table_fixedhead(args.datasets, log_path)

    global _REPLAY_MODE
    workers = _replay_workers()
    if workers > 1:
        _REPLAY_MODE = "collect"
        _run_tables()
        specs = list(_COLLECTED)
        print(
            f"[parallel] prefetching {len(specs)} cells with {workers} processes "
            f"(tune with PAPER_REPLAY_WORKERS; numerically bit-identical to serial)"
        )
        import concurrent.futures as _cf
        import multiprocessing as _mp

        budget_s = float(os.environ.get("PAPER_REPLAY_TIMEOUT_S", "1800"))
        pool = _cf.ProcessPoolExecutor(
            max_workers=workers,
            mp_context=_mp.get_context("spawn"),
            initializer=_worker_init,
            initargs=(FORCE_CPU,),
        )
        futures = {pool.submit(_prefetch_worker, spec): spec for spec in specs}
        done = 0
        try:
            for future in _cf.as_completed(futures, timeout=budget_s):
                spec, metrics, worker_missing = future.result()
                _PREFETCHED[spec] = metrics
                MISSING_ITEMS.update(worker_missing)
                done += 1
                kw = json.loads(spec)
                _progress_tick(
                    f"{kw.get('backbone', '?')} @ {kw.get('dataset', '?')}",
                    done=done,
                    total=len(specs),
                )
        except (_cf.TimeoutError, _cf.process.BrokenProcessPool, OSError) as exc:
            print(
                f"[parallel] prefetch interrupted ({type(exc).__name__}) after "
                f"{done}/{len(specs)} cells; the rest fall back to serial",
                flush=True,
            )
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
        _REPLAY_MODE = "serve"
    _run_tables()
    if {"main", "backbones", "ablation"} <= set(args.tables):
        _emit_paper_csvs(log_path)

    if MISSING_ITEMS:
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write("\n# Missing cells\n")
            for context, reason in sorted(MISSING_ITEMS.items()):
                handle.write(f"- {context}: {reason}\n")
        print(
            f"\nCompleted with {len(MISSING_ITEMS)} missing asset/configuration "
            f"items; affected cells are '{MISSING_CELL}'. Details: {log_path}"
        )
        raise SystemExit(1)
    else:
        print("\nAll requested paper tables replayed from assets/.")


if __name__ == "__main__":
    main()
