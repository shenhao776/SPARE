import os

_VOLATILE_PREFIXES = (
    "ABLATION_",
    "DATA_ABLATION_",
    "DL_",
    "CROP_",
    "SUCCESS_THRESHOLD_",
)

_VOLATILE_NAMES = frozenset(
    {
        "TARGET_SCENE",
        "DATASET",
        "MODEL",
        "VPR_MODEL_TYPE",
        "STEP_MAX_DIST_OVERRIDE",
        "STEP_MAX_DIST_TAG",
        "TRAIN_SEED",
        "USE_MASTER_WEIGHT",
        "ENABLE_SCENE_MODELS_CACHE",
        "OFFLINE_QUERY_CACHE",
        "FEATURE_MATCH_MIN_RAW",
        "PSEUDO_AUG",
        "STAR_GATE_MARGIN",
        "GATE_WEIGHTS",
        "GATE_EXCLUDE",
        "BUILD_SHARD",
        "BUILD_RESUME_CHECK",
        "FORCE_RETRAIN",
        "FORCE_REBUILD_CACHE",
        "FORCE_REBUILD_SEQUENCE_ASSETS",
        "SAVE_IMG",
        "ENABLE_SL",
        "USE_GENERAL_MODEL",
    }
)


def is_volatile(name):
    return name in _VOLATILE_NAMES or name.startswith(_VOLATILE_PREFIXES)


def scrub_environment(env=None):

    env = os.environ if env is None else env
    dropped = {k: env[k] for k in sorted(env) if is_volatile(k)}
    for k in dropped:
        del env[k]
    return dropped


def ensure_reference_scene():

    import os

    if os.environ.get("DATASET") or os.environ.get("TARGET_SCENE"):
        return os.environ.get("DATASET") or os.environ.get("TARGET_SCENE")
    import yaml

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(
        os.path.join(root, "configs", "paper_tables.yaml"), encoding="utf-8"
    ) as fh:
        first_scene = next(iter(yaml.safe_load(fh)["datasets"]))
    ds_dir = os.path.join(root, "configs", "datasets")
    stem = None
    for name in sorted(os.listdir(ds_dir)):
        if not name.endswith(".yaml"):
            continue
        with open(os.path.join(ds_dir, name), encoding="utf-8") as fh:
            if (yaml.safe_load(fh) or {}).get("name") == first_scene:
                stem = os.path.splitext(name)[0]
                break
    if stem is None:
        raise SystemExit(
            f"ensure_reference_scene: 论文配置的第一个场景 {first_scene!r} "
            "在 configs/datasets/ 里找不到对应的 yaml"
        )
    os.environ["DATASET"] = stem
    return stem
