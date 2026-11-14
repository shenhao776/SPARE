import os
import os as _os, sys as _sys

_ROOT = _os.path.dirname(_os.path.abspath(__file__))
if not _os.path.isdir(_os.path.join(_ROOT, "configs")):
    _ROOT = _os.path.dirname(_ROOT)
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
_HERE = _os.path.dirname(_os.path.abspath(__file__))
if _HERE not in _sys.path:
    _sys.path.insert(1, _HERE)

import json
import hashlib
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import random
import numpy as np

from core.config import (
    data_tag,
    DINO_K,
    DL_BATCH_SIZE,
    DL_HIDDEN_DIM,
    DL_EPOCHS,
    DL_LEARNING_RATE,
    SPATIAL_STEPS,
    SUCCESS_THRESHOLD_TRAIN,
    SUCCESS_THRESHOLD_TEST,
    agent_input_dim,
    agent_tag,
    TARGET_SCENE,
    VPR_MODEL_ID,
    DL_MODEL_SAVE_PATH as BASE_MODEL_SAVE_PATH,
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
    ABLATION_USE_AREA,
    ABLATION_USE_CORE,
    ABLATION_USE_SCALE,
    ABLATION_INPUT_NORM,
    ABLATION_ARCH,
    DATA_ABLATION_RANDOM_WINDOW,
)

from sl_agent.vpr_classifier import VPRSequenceDataset, VPRReranking, head_steps
from core.paper_config import load_paper_config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [Reasoning] %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

TRAINING_CACHE_PROTOCOL = 1
DEFAULT_TRAIN_SEED = int(load_paper_config()["seeds"][0])


def _training_fingerprint(train_path, val_path, settings):

    from sl_agent.cache_fingerprint import read_fingerprint

    inputs = []
    for path in (train_path, val_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Training cache input is missing: {path}")
        stamp = read_fingerprint(path) or {}
        cache_key = stamp.get("cache_key") or stamp.get("semantic_digest")
        if not stamp.get("complete") or not cache_key:
            raise ValueError(
                f"训练缓存没有完整的新格式指纹: {path}\n"
                "  旧 JSON 不能补盖章；必须由当前 builder 重建。"
            )
        injectors = dict(stamp.get("injectors") or {})
        tag = str(settings.get("agent_tag") or "")
        if "_vprpatch=True" not in tag and "_vprpatchshuf=True" not in tag:
            injectors.pop("vprpatch", None)
        inputs.append(
            {
                "cache_key": cache_key,
                "injectors": injectors,
            }
        )
    payload = {"inputs": inputs, "settings": settings}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _read_model_meta(path):
    try:
        with open(path + ".meta.json", "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return {}


def _write_model_meta(path, payload):
    meta_path = path + ".meta.json"
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    tmp = f"{meta_path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(tmp, meta_path)


def _model_output_matches(path, meta):

    return bool(isinstance(meta, dict) and os.path.isfile(path))


def _legacy_weight_reusable(meta, train_path, val_path, settings):

    old_settings = dict(meta.get("settings") or {})
    old_settings.pop("content_version", None)
    if settings.get("dino_k") is not None and "dino_k" not in old_settings:
        return False
    if not all(settings.get(key) == value for key, value in old_settings.items()):
        return False
    from sl_agent.cache_fingerprint import read_fingerprint

    stamps = [read_fingerprint(path) or {} for path in (train_path, val_path)]
    return all(
        stamp.get("migrated_from")
        in {
            "ast-output-sha256",
            "legacy-v1-minimal",
            "preprocessing-digest-key",
        }
        for stamp in stamps
    )


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_json_data(json_path):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Dataset not found: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _require_temporal_rows(rows, source, expected_steps):
    temporal_fields = (
        "inliers",
        "scores",
        "step_errors",
        "area_step",
        "area_query",
        "scale_ratio",
    )
    for index, row in enumerate(rows):
        bad = [
            f"{field}={len(row.get(field, []))}"
            for field in temporal_fields
            if len(row.get(field, [])) != expected_steps
        ]
        if bad:
            raise ValueError(
                f"{source} row {index} is not a {expected_steps}-step trajectory: "
                + ", ".join(bad)
            )


def main():
    train_seed = int(os.getenv("TRAIN_SEED", str(DEFAULT_TRAIN_SEED)))
    set_seed(train_seed)
    seed_suffix = f"_seed{train_seed}" if train_seed != DEFAULT_TRAIN_SEED else ""

    dynamic_input_dim = agent_input_dim()
    ablation_str = agent_tag()

    data_ablation_suffix = data_tag()

    scene_suffix = f"_{TARGET_SCENE}" if TARGET_SCENE else "_ALL_SCENES"
    union_tag = os.getenv("TRAIN_SCENE_TAG", "").strip()
    if union_tag:
        if not all(ch.isalnum() or ch in "-_" for ch in union_tag):
            raise SystemExit(
                "TRAIN_SCENE_TAG may contain only letters, digits, '-' and '_'"
            )
        scene_suffix = f"_{union_tag}"

    train_json_path = f"assets/{VPR_MODEL_ID}/train_trajectories{scene_suffix}{data_ablation_suffix}.json"
    val_json_path = f"assets/{VPR_MODEL_ID}/val_trajectories{scene_suffix}{data_ablation_suffix}.json"
    fingerprint_settings = {
        "protocol": TRAINING_CACHE_PROTOCOL,
        "seed": train_seed,
        "input_dim": dynamic_input_dim,
        "hidden_dim": DL_HIDDEN_DIM,
        "epochs": DL_EPOCHS,
        "batch_size": DL_BATCH_SIZE,
        "learning_rate": DL_LEARNING_RATE,
        "spatial_steps": list(SPATIAL_STEPS),
        "success_threshold_train": SUCCESS_THRESHOLD_TRAIN,
        "success_threshold_test": SUCCESS_THRESHOLD_TEST,
        "agent_tag": ablation_str,
        "data_tag": data_ablation_suffix,
        "dino_k": DINO_K if ABLATION_USE_DINO else None,
    }
    training_fingerprint = _training_fingerprint(
        train_json_path, val_json_path, fingerprint_settings
    )

    if os.getenv("ENABLE_SCENE_MODELS_CACHE") == "True":
        model_save_path = BASE_MODEL_SAVE_PATH.replace(
            ".pth",
            f"_{ablation_str}{scene_suffix}{data_ablation_suffix}{seed_suffix}.pth",
        )
        if os.path.exists(model_save_path) and os.getenv("FORCE_RETRAIN") != "True":
            model_meta = _read_model_meta(model_save_path)
            if (
                model_meta.get("status") == "complete"
                and model_meta.get("training_fingerprint") == training_fingerprint
                and _model_output_matches(model_save_path, model_meta)
            ):
                logger.info(
                    f"⏩ [Cache] Weight and train/val fingerprints match: "
                    f"{os.path.basename(model_save_path)}. Skipping training."
                )
                return
            if (
                model_meta.get("status") == "complete"
                and _model_output_matches(model_save_path, model_meta)
                and _legacy_weight_reusable(
                    model_meta,
                    train_json_path,
                    val_json_path,
                    fingerprint_settings,
                )
            ):
                _write_model_meta(
                    model_save_path,
                    {
                        **model_meta,
                        "training_fingerprint": training_fingerprint,
                        "settings": fingerprint_settings,
                        "migrated_from": "ast-output-sha256",
                    },
                )
                logger.info(
                    f"⏩ [Cache] Existing weight inputs are unchanged; migrated "
                    f"the small sidecar and skipped training: "
                    f"{os.path.basename(model_save_path)}"
                )
                return
            logger.info(
                f"♻️ [Rebuild] Existing weight has no matching train/val "
                f"fingerprint: {os.path.basename(model_save_path)}"
            )
    else:
        model_save_path = BASE_MODEL_SAVE_PATH.replace(
            ".pth", f"_{ablation_str}{data_ablation_suffix}{seed_suffix}.pth"
        )

    if os.getenv("TRAIN_RESUME_CHECK") == "1":
        logger.info(f"RESUME_REBUILD: {os.path.basename(model_save_path)} would train")
        raise SystemExit(10)

    logger.info("======================================================")
    logger.info(f"🤖 Starting ablation run: {ablation_str}")
    logger.info(f"📐 Feature dimension: {dynamic_input_dim}D")

    logger.info(
        f"🗂️  Data ablation: {data_ablation_suffix if data_ablation_suffix else 'Normal (Full Data)'}"
    )

    train_raw_data = load_json_data(train_json_path)
    val_raw_data = load_json_data(val_json_path)
    expected_steps = sum(SPATIAL_STEPS) + 1
    _require_temporal_rows(train_raw_data, train_json_path, expected_steps)
    _require_temporal_rows(val_raw_data, val_json_path, expected_steps)
    head_window = head_steps(ABLATION_ARCH, expected_steps)
    logger.info(
        f"🪟 Head window: {head_window}/{expected_steps} steps"
        + (" (centre frame only)" if head_window == 1 else "")
    )

    train_dataset = VPRSequenceDataset(
        train_raw_data,
        spatial_steps=SPATIAL_STEPS,
        use_derivatives=ABLATION_USE_DERIVATIVES,
        use_progress=ABLATION_USE_PROGRESS,
        use_sim=ABLATION_USE_SIM,
        use_area=ABLATION_USE_AREA,
        use_rank=ABLATION_USE_RANK,
        use_dino=ABLATION_USE_DINO,
        use_pose=ABLATION_USE_POSE,
        use_hull=ABLATION_USE_HULL,
        use_homog=ABLATION_USE_HOMOG,
        use_walk=ABLATION_USE_WALK,
        use_shape=ABLATION_USE_SHAPE,
        use_zoom=ABLATION_USE_ZOOM,
        use_zoomshuf=ABLATION_USE_ZOOMSHUF,
        arch=ABLATION_ARCH,
        use_core=ABLATION_USE_CORE,
        use_scale=ABLATION_USE_SCALE,
    )
    val_dataset = VPRSequenceDataset(
        val_raw_data,
        spatial_steps=SPATIAL_STEPS,
        use_derivatives=ABLATION_USE_DERIVATIVES,
        use_progress=ABLATION_USE_PROGRESS,
        use_sim=ABLATION_USE_SIM,
        use_area=ABLATION_USE_AREA,
        use_rank=ABLATION_USE_RANK,
        use_dino=ABLATION_USE_DINO,
        use_pose=ABLATION_USE_POSE,
        use_hull=ABLATION_USE_HULL,
        use_homog=ABLATION_USE_HOMOG,
        use_walk=ABLATION_USE_WALK,
        use_shape=ABLATION_USE_SHAPE,
        use_zoom=ABLATION_USE_ZOOM,
        use_zoomshuf=ABLATION_USE_ZOOMSHUF,
        arch=ABLATION_ARCH,
        use_core=ABLATION_USE_CORE,
        use_scale=ABLATION_USE_SCALE,
    )

    train_loader = DataLoader(train_dataset, batch_size=DL_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=DL_BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VPRReranking(
        input_dim=dynamic_input_dim,
        hidden_dim=DL_HIDDEN_DIM,
        arch=ABLATION_ARCH,
        input_norm=ABLATION_INPUT_NORM,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=DL_LEARNING_RATE, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=DL_EPOCHS, eta_min=1e-6)

    best_val_acc = 0.0
    best_val_loss = float("inf")
    final_val_acc = 0.0

    _write_model_meta(
        model_save_path,
        {
            "status": "building",
            "training_fingerprint": training_fingerprint,
            "settings": fingerprint_settings,
        },
    )

    for epoch in range(DL_EPOCHS):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y, batch_mask, _ in train_loader:
            batch_x, batch_y, batch_mask = (
                batch_x.to(device),
                batch_y.to(device),
                batch_mask.to(device),
            )
            optimizer.zero_grad()
            logits = model(batch_x, mask=batch_mask)

            loss, weight_sum = 0.0, 0.0
            seq_len = logits.size(2)

            for t in range(seq_len):
                step_weight = (t + 1) / seq_len
                loss += criterion(logits[:, :, t], batch_y) * step_weight
                weight_sum += step_weight

            loss = loss / weight_sum
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)

        train_loss /= len(train_loader.dataset)
        scheduler.step()

        model.eval()
        val_loss, correct_count = 0.0, 0
        with torch.no_grad():
            for batch_x, batch_y, batch_mask, _ in val_loader:
                batch_x, batch_y, batch_mask = (
                    batch_x.to(device),
                    batch_y.to(device),
                    batch_mask.to(device),
                )
                logits = model(batch_x, mask=batch_mask)

                step_loss, weight_sum = 0.0, 0.0
                seq_len = logits.size(2)
                for t in range(seq_len):
                    step_weight = (t + 1) / seq_len
                    step_loss += criterion(logits[:, :, t], batch_y) * step_weight
                    weight_sum += step_weight

                val_loss += (step_loss / weight_sum).item() * batch_x.size(0)
                predictions = torch.argmax(logits[:, :, -1], dim=1)

                for b in range(batch_x.size(0)):
                    if (torch.max(batch_y[b]) - torch.min(batch_y[b])).item() >= 1e-4:
                        if batch_y[b, predictions[b].item()].item() > 0:
                            correct_count += 1

            val_acc = (correct_count / val_dataset.total_original_queries) * 100
        val_loss /= len(val_loader.dataset)

        logger.info(
            f"[{TARGET_SCENE}|seed{train_seed}] Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%"
        )

        if val_acc > best_val_acc or (
            val_acc == best_val_acc and val_loss < best_val_loss
        ):
            best_val_acc, best_val_loss, final_val_acc = val_acc, val_loss, val_acc
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            tmp_model_path = f"{model_save_path}.tmp.{os.getpid()}"
            torch.save(model.state_dict(), tmp_model_path)
            os.replace(tmp_model_path, model_save_path)

    _write_model_meta(
        model_save_path,
        {
            "status": "complete",
            "training_fingerprint": training_fingerprint,
            "settings": fingerprint_settings,
            "best_val_accuracy": final_val_acc,
        },
    )

    logger.info(
        f"Training finished [{ablation_str} | Data: {data_ablation_suffix if data_ablation_suffix else 'Full'}]! Best val acc: {final_val_acc:.2f}%"
    )


if __name__ == "__main__":
    main()
