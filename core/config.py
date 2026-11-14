import os
import sys
import logging
from types import SimpleNamespace

import yaml

current_file_path = os.path.abspath(__file__)
ROOT_DIR = os.path.dirname(os.path.dirname(current_file_path))
CONFIGS_DIR = os.path.join(ROOT_DIR, "configs")


def _load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve(kind, selector):

    base = os.path.join(CONFIGS_DIR, kind)
    for stem in dict.fromkeys([selector, selector.lower()]):
        path = os.path.join(base, f"{stem}.yaml")
        if os.path.exists(path):
            return stem, _load_yaml(path)
    for fn in sorted(os.listdir(base)):
        if not fn.endswith(".yaml"):
            continue
        cfg = _load_yaml(os.path.join(base, fn))
        if str(cfg.get("name", "")).lower() == selector.lower():
            return fn[:-5], cfg
    raise FileNotFoundError(f"No {kind} config matching '{selector}' under {base}")


def _ns(obj):

    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_ns(v) for v in obj]
    return obj


def resolve_dataset(selector):

    return _resolve("datasets", selector)[1]


def _envbool(name, default_bool):
    return os.getenv(name, "True" if default_bool else "False") == "True"


_DEFAULTS = _load_yaml(os.path.join(CONFIGS_DIR, "default.yaml"))

_dataset_sel = os.environ.get("DATASET") or os.environ.get("TARGET_SCENE")
if not _dataset_sel:
    _stems = sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(os.path.join(CONFIGS_DIR, "datasets"))
        if f.endswith(".yaml")
    )
    raise SystemExit(
        "core.config: 必须指定场景 —— 设 DATASET=<stem>(或旧写法 TARGET_SCENE)。\n"
        "  不指定时不会回退到某个默认场景,否则裁剪带/阈值/step_budget 会取自别的场景。\n"
        f"  可选 stem:{', '.join(_stems)}"
    )
_model_sel = (
    os.environ.get("MODEL")
    or os.environ.get("VPR_MODEL_TYPE")
    or _DEFAULTS["defaults"]["model"]
)

_ds_stem, _ds = _resolve("datasets", _dataset_sel)
_md_stem, _md = _resolve("models", _model_sel)


TARGET_SCENE = _ds["name"]
VPR_MODEL_TYPE = _md["name"]

ENABLE_SL = bool(_DEFAULTS["enable_sl"])
USE_GENERAL_MODEL = bool(_DEFAULTS["use_general_model"])

TOP_K = int(_DEFAULTS["top_k"])

DINO_K = int(_DEFAULTS["dino_k"])
if DINO_K < 1 or DINO_K % 2 == 0:
    raise SystemExit(f"dino_k 必须是正奇数(中心±(k-1)/2),当前 {DINO_K}")

SAVE_IMG = bool(_DEFAULTS["save_img"])

SERVER_PORT = int(_DEFAULTS["server"]["port"])
BUFFER_SIZE = int(_DEFAULTS["server"]["buffer_size"])

_seq = _ds.get("sequence") or {}
_scene_yaml = f"configs/datasets/{_ds_stem}.yaml"


def _need(key, why):
    if _seq.get(key) is None:
        raise SystemExit(
            f"[{TARGET_SCENE}] {_scene_yaml} 的 sequence 段缺 `{key}`。\n"
            f"  {why}\n"
            f"  没有默认值可退 —— 默认值会让忘记声明这件事不产生任何症状。"
        )
    return _seq[key]


SPATIAL_STEPS = list(
    _need("spatial_steps", "窗口宽度上限,如 [10, 10] = 21 步。它是硬上限,不是步长。")
)

STEP_FEATURE_TAU = float(
    _need(
        "step_feature_tau",
        "每积累这么多外观变化就采一步。变化量 = 1 - 相邻帧单应把画框warp过去之后的 IoU。",
    )
)
STEP_MAX_DIST = float(
    os.getenv(
        "STEP_MAX_DIST_OVERRIDE",
        _need(
            "step_max_dist",
            "采样天花板:走这么多米还没变化也采一步。固定协议读 YAML；受控 A/B "
            "可由 reproduce.py 显式注入 mapping-only 标定值。",
        ),
    )
)
STEP_MAX_DIST_TAG = os.getenv("STEP_MAX_DIST_TAG", "").strip()
if STEP_MAX_DIST_TAG and not all(
    ch.isalnum() or ch in "-_" for ch in STEP_MAX_DIST_TAG
):
    raise SystemExit("STEP_MAX_DIST_TAG may contain only letters, digits, '-' and '_'")
STEP_BUDGET = float(
    _need(
        "step_budget",
        "停止判据,**由 tools/map_feature_rate.py 标定**,不能照抄别的场景:饱和值逐图"
        "不同(实测 Boreas 0.271 / BusinessCampus 0.514 / NCLT 0.86)。\n"
        "  `python reproduce.py --paper-order <该场景>` 会自动标定并写回 yaml。",
    )
)

GEOMETRY_KEYS = (
    "hull_cx",
    "hull_cy",
    "hull_aspect",
    "hull_fill",
    "H_tx",
    "H_ty",
    "H_logdet",
    "H_persp",
)
GEOMETRY_VERSION = 2

BASE_GROUPS = {
    "core": "ABLATION_USE_CORE",
    "scale": "ABLATION_USE_SCALE",
    "innorm": "ABLATION_INPUT_NORM",
    "deriv": "ABLATION_USE_DERIVATIVES",
    "progress": "ABLATION_USE_PROGRESS",
    "sim": "ABLATION_USE_SIM",
    "area": "ABLATION_USE_AREA",
    "rank": "ABLATION_USE_RANK",
}

CHANNEL_GROUPS = {
    "pose": "ABLATION_USE_POSE",
    "hull": "ABLATION_USE_HULL",
    "homog": "ABLATION_USE_HOMOG",
    "walk": "ABLATION_USE_WALK",
    "shape": "ABLATION_USE_SHAPE",
    "dino": "ABLATION_USE_DINO",
    "zoom": "ABLATION_USE_ZOOM",
    "zoomshuf": "ABLATION_USE_ZOOMSHUF",
    "vprpatch": "ABLATION_USE_VPRPATCH",
    "vprpatchshuf": "ABLATION_USE_VPRPATCHSHUF",
}


_ds_th = {**_DEFAULTS.get("thresholds", {}), **(_ds.get("thresholds") or {})}
SUCCESS_THRESHOLD_TRAIN = float(os.getenv("SUCCESS_THRESHOLD_TRAIN", _ds_th["train_m"]))
SUCCESS_THRESHOLD_TEST = float(os.getenv("SUCCESS_THRESHOLD_TEST", _ds_th["test_m"]))
THRESHOLD_TEST_DEFAULT = float(_ds_th["test_m"])
MAX_ACCEPTABLE_ERROR = float(_DEFAULTS["max_acceptable_error"])
MIN_CANDIDATE_DIST = float(_DEFAULTS["min_candidate_dist"])

VPR_BACKBONE = "ResNet50"
VPR_FC_OUTPUT_DIM = int(_md["feature_dim"])
VPR_MODEL_ID = _md["model_id"]

if os.getenv("DL_MODEL_SAVE_PATH"):
    DL_MODEL_SAVE_PATH = os.getenv("DL_MODEL_SAVE_PATH")
elif USE_GENERAL_MODEL:
    DL_MODEL_SAVE_PATH = os.path.join(
        ROOT_DIR, "assets", "general_model", "vpr_general.pth"
    )
else:
    DL_MODEL_SAVE_PATH = os.path.join(
        ROOT_DIR, "assets", VPR_MODEL_ID, "models", "vpr_reranking_gru.pth"
    )

_rr = _DEFAULTS["reranking"]
DL_LEARNING_RATE = float(os.getenv("DL_LEARNING_RATE", _rr["learning_rate"]))
DL_BATCH_SIZE = int(os.getenv("DL_BATCH_SIZE", _rr["batch_size"]))
DL_EPOCHS = int(os.getenv("DL_EPOCHS", _rr["epochs"]))
DL_HIDDEN_DIM = int(os.getenv("DL_HIDDEN_DIM", _rr["hidden_dim"]))

_geo = _DEFAULTS["geometry"]
FEATURE_MATCH_MIN_RAW = int(
    os.getenv("FEATURE_MATCH_MIN_RAW", _geo.get("feature_match_min_raw", 10))
)
FEATURE_MATCH_MIN_INLIERS = int(_geo["feature_match_min_inliers"])
FEATURE_HULL_MIN_INLIERS = int(_geo["feature_hull_min_inliers"])
RANSAC_REPROJ_THRESHOLD = float(_geo["ransac_reproj_threshold"])
LOG1P = bool(_geo["log1p"])
INLIER_NORM_FACTOR = float(_geo["inlier_norm_factor"])

_abl = _DEFAULTS.get("ablation", {})
ABLATION_USE_DERIVATIVES = _envbool(
    "ABLATION_USE_DERIVATIVES", _abl.get("use_derivatives", False)
)
ABLATION_USE_PROGRESS = _envbool(
    "ABLATION_USE_PROGRESS", _abl.get("use_progress", True)
)
ABLATION_USE_CORE = _envbool("ABLATION_USE_CORE", _abl.get("use_core", True))
ABLATION_USE_SCALE = _envbool("ABLATION_USE_SCALE", _abl.get("use_scale", True))
ABLATION_INPUT_NORM = _envbool("ABLATION_INPUT_NORM", _abl.get("input_norm", True))
ABLATION_ARCH = os.getenv("ABLATION_ARCH", _abl.get("arch", "gru"))
DATA_ABLATION_RANDOM_WINDOW = _envbool(
    "DATA_ABLATION_RANDOM_WINDOW", _abl.get("data_random_window", False)
)

SEQUENCE_ASSET_FORMAT = "lean-v2"
ABLATION_USE_SIM = _envbool("ABLATION_USE_SIM", _abl.get("use_sim", False))
ABLATION_USE_AREA = _envbool("ABLATION_USE_AREA", _abl.get("use_area", False))
ABLATION_USE_RANK = _envbool("ABLATION_USE_RANK", _abl.get("use_rank", False))
ABLATION_USE_DINO = _envbool("ABLATION_USE_DINO", _abl.get("use_dino", False))
ABLATION_USE_POSE = _envbool("ABLATION_USE_POSE", _abl.get("use_pose", False))
ABLATION_USE_HULL = _envbool("ABLATION_USE_HULL", _abl.get("use_hull", False))
ABLATION_USE_HOMOG = _envbool("ABLATION_USE_HOMOG", _abl.get("use_homog", False))
ABLATION_USE_WALK = _envbool("ABLATION_USE_WALK", _abl.get("use_walk", False))
ABLATION_USE_SHAPE = _envbool("ABLATION_USE_SHAPE", _abl.get("use_shape", False))
ABLATION_USE_ZOOM = _envbool("ABLATION_USE_ZOOM", _abl.get("use_zoom", False))
ABLATION_USE_ZOOMSHUF = _envbool(
    "ABLATION_USE_ZOOMSHUF", _abl.get("use_zoomshuf", False)
)
ABLATION_USE_VPRPATCH = _envbool(
    "ABLATION_USE_VPRPATCH", _abl.get("use_vprpatch", False)
)
ABLATION_USE_VPRPATCHSHUF = _envbool(
    "ABLATION_USE_VPRPATCHSHUF", _abl.get("use_vprpatchshuf", False)
)


def agent_tag(
    arch=None,
    derivatives=None,
    progress=None,
    use_sim=None,
    use_area=None,
    use_rank=None,
    use_dino=None,
    use_pose=None,
    use_hull=None,
    use_homog=None,
    use_walk=None,
    use_shape=None,
    use_zoom=None,
    use_zoomshuf=None,
    use_core=None,
    use_scale=None,
    input_norm=None,
    use_vprpatch=None,
    use_vprpatchshuf=None,
):

    arch = ABLATION_ARCH if arch is None else arch
    derivatives = ABLATION_USE_DERIVATIVES if derivatives is None else derivatives
    progress = ABLATION_USE_PROGRESS if progress is None else progress
    use_sim = ABLATION_USE_SIM if use_sim is None else use_sim
    use_area = ABLATION_USE_AREA if use_area is None else use_area
    use_rank = ABLATION_USE_RANK if use_rank is None else use_rank
    use_dino = ABLATION_USE_DINO if use_dino is None else use_dino
    use_pose = ABLATION_USE_POSE if use_pose is None else use_pose
    use_hull = ABLATION_USE_HULL if use_hull is None else use_hull
    use_homog = ABLATION_USE_HOMOG if use_homog is None else use_homog
    use_walk = ABLATION_USE_WALK if use_walk is None else use_walk
    use_shape = ABLATION_USE_SHAPE if use_shape is None else use_shape
    use_zoom = ABLATION_USE_ZOOM if use_zoom is None else use_zoom
    use_zoomshuf = ABLATION_USE_ZOOMSHUF if use_zoomshuf is None else use_zoomshuf
    use_core = ABLATION_USE_CORE if use_core is None else use_core
    use_scale = ABLATION_USE_SCALE if use_scale is None else use_scale
    input_norm = ABLATION_INPUT_NORM if input_norm is None else input_norm
    use_vprpatch = ABLATION_USE_VPRPATCH if use_vprpatch is None else use_vprpatch
    use_vprpatchshuf = (
        ABLATION_USE_VPRPATCHSHUF if use_vprpatchshuf is None else use_vprpatchshuf
    )
    tag = f"arch={arch}_deriv={derivatives}_prog={progress}"
    if not use_core:
        tag += "_core=False"
    elif not use_scale:
        tag += "_scale=False"
    if use_sim:
        tag += "_sim=True"
    if use_area:
        tag += "_area=True"
    if use_rank:
        tag += "_rank=True"
    if use_dino:
        tag += "_dino=True"
    if use_pose:
        tag += "_pose=True"
    if use_hull:
        tag += "_hull=True"
    if use_homog:
        tag += "_homog=True"
    if use_walk:
        tag += "_walk=True"
    if use_shape:
        tag += "_shape=True"
    if use_zoom:
        tag += "_zoom=True"
    if use_zoomshuf:
        tag += "_zoomshuf=True"
    if use_vprpatch:
        tag += "_vprpatch=True"
    if use_vprpatchshuf:
        tag += "_vprpatchshuf=True"
    if not input_norm:
        tag += "_innorm=False"
    if abs(SUCCESS_THRESHOLD_TEST - THRESHOLD_TEST_DEFAULT) > 1e-9:
        tag += f"_th{SUCCESS_THRESHOLD_TEST:g}"
    return tag


def data_tag(random_window=None, step_max_tag=None):

    random_window = (
        DATA_ABLATION_RANDOM_WINDOW if random_window is None else random_window
    )
    step_max_tag = STEP_MAX_DIST_TAG if step_max_tag is None else step_max_tag
    suffix = "_random_window" if random_window else ""
    if step_max_tag:
        suffix += f"_stepmax-{step_max_tag}"
    return suffix + "_no_roi"


def paper_protocol_env():

    return {
        "ABLATION_ARCH": ABLATION_ARCH,
        "ABLATION_USE_CORE": str(ABLATION_USE_CORE),
        "ABLATION_USE_SCALE": str(ABLATION_USE_SCALE),
        "ABLATION_INPUT_NORM": str(ABLATION_INPUT_NORM),
        "ABLATION_USE_DERIVATIVES": str(ABLATION_USE_DERIVATIVES),
        "ABLATION_USE_PROGRESS": str(ABLATION_USE_PROGRESS),
        "ABLATION_USE_SIM": str(ABLATION_USE_SIM),
        "ABLATION_USE_AREA": str(ABLATION_USE_AREA),
        "ABLATION_USE_RANK": str(ABLATION_USE_RANK),
        "ABLATION_USE_DINO": str(ABLATION_USE_DINO),
        "ABLATION_USE_POSE": str(ABLATION_USE_POSE),
        "ABLATION_USE_HULL": str(ABLATION_USE_HULL),
        "ABLATION_USE_HOMOG": str(ABLATION_USE_HOMOG),
        "ABLATION_USE_WALK": str(ABLATION_USE_WALK),
        "ABLATION_USE_SHAPE": str(ABLATION_USE_SHAPE),
        "ABLATION_USE_ZOOM": str(ABLATION_USE_ZOOM),
        "ABLATION_USE_ZOOMSHUF": str(ABLATION_USE_ZOOMSHUF),
        "ABLATION_USE_VPRPATCH": str(ABLATION_USE_VPRPATCH),
        "ABLATION_USE_VPRPATCHSHUF": str(ABLATION_USE_VPRPATCHSHUF),
        "DATA_ABLATION_RANDOM_WINDOW": str(DATA_ABLATION_RANDOM_WINDOW),
    }


def agent_input_dim(
    derivatives=None,
    progress=None,
    use_sim=None,
    use_area=None,
    use_rank=None,
    use_dino=None,
    use_pose=None,
    use_hull=None,
    use_homog=None,
    use_walk=None,
    use_shape=None,
    use_zoom=None,
    use_zoomshuf=None,
    use_core=None,
    use_scale=None,
    use_vprpatch=None,
    use_vprpatchshuf=None,
):

    derivatives = ABLATION_USE_DERIVATIVES if derivatives is None else derivatives
    progress = ABLATION_USE_PROGRESS if progress is None else progress
    use_sim = ABLATION_USE_SIM if use_sim is None else use_sim
    use_area = ABLATION_USE_AREA if use_area is None else use_area
    use_rank = ABLATION_USE_RANK if use_rank is None else use_rank
    use_dino = ABLATION_USE_DINO if use_dino is None else use_dino
    use_pose = ABLATION_USE_POSE if use_pose is None else use_pose
    use_hull = ABLATION_USE_HULL if use_hull is None else use_hull
    use_homog = ABLATION_USE_HOMOG if use_homog is None else use_homog
    use_walk = ABLATION_USE_WALK if use_walk is None else use_walk
    use_shape = ABLATION_USE_SHAPE if use_shape is None else use_shape
    use_zoom = ABLATION_USE_ZOOM if use_zoom is None else use_zoom
    use_zoomshuf = ABLATION_USE_ZOOMSHUF if use_zoomshuf is None else use_zoomshuf
    use_core = ABLATION_USE_CORE if use_core is None else use_core
    use_scale = ABLATION_USE_SCALE if use_scale is None else use_scale
    use_vprpatch = ABLATION_USE_VPRPATCH if use_vprpatch is None else use_vprpatch
    use_vprpatchshuf = (
        ABLATION_USE_VPRPATCHSHUF if use_vprpatchshuf is None else use_vprpatchshuf
    )
    return (
        ((2 if use_scale else 1) if use_core else 0)
        + (2 if derivatives else 0)
        + (1 if progress else 0)
        + (1 if use_sim else 0)
        + (2 if use_area else 0)
        + (1 if use_rank else 0)
        + (1 if use_dino else 0)
        + (2 if use_pose else 0)
        + (4 if use_hull else 0)
        + (4 if use_homog else 0)
        + (2 if use_walk else 0)
        + (2 if use_shape else 0)
        + (1 if use_zoom else 0)
        + (1 if use_zoomshuf else 0)
        + (1 if use_vprpatch else 0)
        + (1 if use_vprpatchshuf else 0)
    )


_ds_resolved = dict(_ds)
_ds_resolved["selector"] = _ds_stem
_ds_resolved["thresholds"] = {
    "train_m": SUCCESS_THRESHOLD_TRAIN,
    "test_m": SUCCESS_THRESHOLD_TEST,
}

_md_resolved = dict(_md)
_md_resolved["selector"] = _md_stem

CFG = SimpleNamespace(
    dataset=_ns(_ds_resolved),
    model=_ns(_md_resolved),
    dataset_dict=_ds_resolved,
    model_dict=_md_resolved,
    root_dir=ROOT_DIR,
    top_k=TOP_K,
    save_img=SAVE_IMG,
    query_crop=_ns(_DEFAULTS["query_crop"]),
    enable_sl=ENABLE_SL,
    spatial_steps=SPATIAL_STEPS,
    step_feature_tau=STEP_FEATURE_TAU,
    step_max_dist=STEP_MAX_DIST,
    step_budget=STEP_BUDGET,
    min_candidate_dist=MIN_CANDIDATE_DIST,
    max_acceptable_error=MAX_ACCEPTABLE_ERROR,
)


LOGS_DIR = os.path.join(ROOT_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)


def setup_vpr_logger(log_filename=None):
    logger = logging.getLogger("VPR_Logger")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)
    if log_filename:
        log_file = os.path.join(LOGS_DIR, log_filename)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(
            logging.Formatter(
                "[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
            )
        )
        logger.addHandler(fh)
    return logger


vpr_logger = setup_vpr_logger(None)
