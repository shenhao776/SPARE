import os as _os, sys as _sys

_ROOT = _os.path.dirname(_os.path.abspath(__file__))
if not _os.path.isdir(_os.path.join(_ROOT, "configs")):
    _ROOT = _os.path.dirname(_ROOT)
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
_HERE = _os.path.dirname(_os.path.abspath(__file__))
if _HERE not in _sys.path:
    _sys.path.insert(1, _HERE)

import argparse
import concurrent.futures
import importlib
import json
import math
import os
import re
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

RUNS_DIR = os.path.join("logs", "runs")
_RUN_LOG = None
_LAST_LOG_LINE = None

_DROP = re.compile(
    r"it/s\]|it\]$|\[SERVER\]|Warning:|Using cache found|"
    r"^\s*$|^\s*=+$|^# [A-Za-z]+  partition:|^--- 🚀|^--- 📦|^--- \*\*\*"
)
_KEEP = re.compile(
    r"Traceback|Error|Exception|FAILED|failed|\[FATAL\]|\[Resume\]|\[Rebuild\]|"
    r"\[parallel\]|\[done\]|\[input ok\]|\[layout|\[upgrade\]|\[skip |"
    r"✅|Extracted|true hits|Training finished|## Table|backbone |paper weight"
)


def _ts():
    now = time.localtime()
    tz = time.strftime("%z", now)
    return time.strftime("%Y-%m-%dT%H:%M:%S", now) + tz


def git_provenance():

    def _git(*args):
        try:
            out = subprocess.run(
                ("git",) + args, cwd=REPO, capture_output=True, text=True, timeout=10
            )
            return out.stdout.strip() if out.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None

    commit = _git("rev-parse", "HEAD") or "unknown"
    status = _git("status", "--porcelain")
    return commit, bool(status)


def init_run_log(dataset):

    global _RUN_LOG
    os.makedirs(RUNS_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    _RUN_LOG = os.path.join(RUNS_DIR, f"reproduce_{dataset or 'all'}_{stamp}.log")
    commit, dirty = git_provenance()
    header = [
        f"[logged] reproduce {dataset or 'all'}",
        f"[provenance] commit {commit}{' +dirty' if dirty else ''} "
        f"host {os.uname().nodename if hasattr(os, 'uname') else '?'}",
    ]
    if _SCRUBBED:
        header.append(
            "[env] dropped inherited protocol variables: "
            + ", ".join(f"{k}={v}" for k, v in _SCRUBBED.items())
        )
    else:
        header.append("[env] no inherited protocol variables")
    try:
        with open(_RUN_LOG, "w", encoding="utf-8") as fh:
            for line in header:
                fh.write(f"{_ts()} {line}\n")
    except OSError:
        _RUN_LOG = None
    for line in header[1:]:
        print(line, flush=True)
    return _RUN_LOG


_EVAL_DETAIL = re.compile(r"^└─ \[")


def _log_line(line):

    global _LAST_LOG_LINE, _LAST_PROGRESS_TS
    if _RUN_LOG is None:
        return
    clean = line.rstrip("\n")
    if not clean.strip():
        return
    if clean == _LAST_LOG_LINE:
        return
    _LAST_LOG_LINE = clean
    if re.search(r"\d+%\|", clean):
        if time.time() - _LAST_PROGRESS_TS < _PROGRESS_LOG_SEC:
            return
        _LAST_PROGRESS_TS = time.time()
    try:
        with open(_RUN_LOG, "a", encoding="utf-8") as fh:
            fh.write(clean + "\n")
    except OSError:
        pass


def _log(text):

    for line in text.splitlines():
        _log_line(line)


_PROGRESS_LOG_SEC = float(os.getenv("STAR_VPR_PROGRESS_LOG_SEC", "60"))
_LAST_PROGRESS_TS = 0.0


_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_OURS_ROW = re.compile(r"\|\s*Ours\s*\([^)]*\)\s*\|\s*\d+/\d+\s*\|\s*([0-9.]+)%")


def _run_log_size():
    try:
        return os.path.getsize(_RUN_LOG) if _RUN_LOG else 0
    except OSError:
        return 0


def _last_ours_pct(since):

    if _RUN_LOG is None:
        return None
    try:
        with open(_RUN_LOG, "r", encoding="utf-8") as fh:
            fh.seek(since)
            tail = fh.read()
    except OSError:
        return None
    hits = _OURS_ROW.findall(_ANSI.sub("", tail))
    return float(hits[-1]) if hits else None


if os.name == "posix":
    _runtime_tmp = os.environ.get("STAR_VPR_TMPDIR", "/tmp")
    if not os.path.isdir(_runtime_tmp):
        raise RuntimeError(f"STAR_VPR_TMPDIR does not exist: {_runtime_tmp}")
    os.environ["TMPDIR"] = _runtime_tmp
    os.environ["TEMP"] = _runtime_tmp
    os.environ["TMP"] = _runtime_tmp

sys.path.insert(0, REPO)
from core.env_isolation import (
    ensure_reference_scene,
    scrub_environment,
)

_SCRUBBED = scrub_environment()

ensure_reference_scene()

from core.config import (
    BASE_GROUPS,
    CHANNEL_GROUPS,
    agent_input_dim,
    paper_protocol_env,
)
from core.paper_config import load_paper_config

FULL_VARIANT_ENV = paper_protocol_env()
PAPER_CONFIG = load_paper_config(FULL_VARIANT_ENV)

SCENE_NAME = {stem: scene for scene, stem in PAPER_CONFIG["all_scene_stems"].items()}
TRAJECTORY_DATASETS = {
    dataset: [PAPER_CONFIG["scene_stems"][scene] for scene in spec["scenes"]]
    for dataset, spec in PAPER_CONFIG["datasets"].items()
}
TRAJECTORY_DATASETS.update({scene: [stem] for stem, scene in SCENE_NAME.items()})
PAPER_DATASETS = list(PAPER_CONFIG["datasets"])
PAPER_BACKBONES = list(PAPER_CONFIG["backbones"])
_MODEL_ID_OF = dict(PAPER_CONFIG["model_ids"])
PAPER_SEEDS = tuple(int(seed) for seed in PAPER_CONFIG["seeds"])
TABLE3_DATASETS = {
    dataset
    for dataset, spec in PAPER_CONFIG["datasets"].items()
    if spec.get("ablation")
}
MAIN_BACKBONE = PAPER_CONFIG["main_backbone"]
BUILD_SPLITS = ()
MODEL_ID = _MODEL_ID_OF[MAIN_BACKBONE]
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

_DIM_KWARG = {
    "ABLATION_USE_CORE": "use_core",
    "ABLATION_USE_SCALE": "use_scale",
    "ABLATION_USE_DERIVATIVES": "derivatives",
    "ABLATION_USE_PROGRESS": "progress",
    "ABLATION_USE_SIM": "use_sim",
    "ABLATION_USE_AREA": "use_area",
    "ABLATION_USE_RANK": "use_rank",
    "ABLATION_USE_DINO": "use_dino",
    "ABLATION_USE_POSE": "use_pose",
    "ABLATION_USE_HULL": "use_hull",
    "ABLATION_USE_HOMOG": "use_homog",
    "ABLATION_USE_WALK": "use_walk",
    "ABLATION_USE_SHAPE": "use_shape",
    "ABLATION_USE_ZOOM": "use_zoom",
    "ABLATION_USE_ZOOMSHUF": "use_zoomshuf",
    "ABLATION_USE_VPRPATCH": "use_vprpatch",
    "ABLATION_USE_VPRPATCHSHUF": "use_vprpatchshuf",
}


def _apply_channels(env, channels_on):

    for group in channels_on:
        if group.startswith("no-"):
            name = group[3:]
            env[(BASE_GROUPS.get(name) or CHANNEL_GROUPS[name])] = "False"
        else:
            env[CHANNEL_GROUPS[group]] = "True"
    return env


def _state_dim_of(channels_on):

    env = _apply_channels(dict(FULL_VARIANT_ENV), channels_on)
    return agent_input_dim(
        **{kw: env.get(var, "False") == "True" for var, kw in _DIM_KWARG.items()}
    )


def _require_runtime_dependencies(stages=None):

    heavy = not stages or bool({"layout", "build"} & set(stages))
    required = {
        "torch": "PyTorch",
        "torchvision": "torchvision",
        "numpy": "NumPy",
        "yaml": "PyYAML",
        "tqdm": "tqdm",
    }
    if heavy:
        required.update(
            {
                "cv2": "OpenCV",
                "faiss": "FAISS",
                "scipy": "SciPy",
                "PIL": "Pillow",
                "einops": "einops",
            }
        )
    errors = []
    for module, package in required.items():
        try:
            importlib.import_module(module)
        except Exception as exc:
            errors.append(f"{package} ({module}): {type(exc).__name__}: {exc}")

    if heavy:
        try:
            from lightglue import LightGlue, SuperPoint
        except Exception as exc:
            errors.append(
                f"LightGlue (lightglue.LightGlue/SuperPoint): "
                f"{type(exc).__name__}: {exc}"
            )

    if errors:
        details = "\n  - ".join(errors)
        raise SystemExit(
            "[dependency error] reproduction environment is incomplete; no stage "
            "was started.\n  - " + details + "\n"
            "Create/update the GPU environment with: "
            "conda env update -n vpr_env -f env/clean_env.yml --prune"
        )
    print("[environment ok] core runtime + LightGlue imports succeeded")


TRANSIENT_GPU_MARKERS = (
    "unspecified launch failure",
    "CUDA error: unknown error",
    "CUDA driver error",
    "cuDNN error: CUDNN_STATUS_EXECUTION_FAILED",
)
TRANSIENT_RETRIES = 2
RETRY_SLEEP_S = 20


_PROG = {}


def _prog(key, i, total, name):
    _PROG[key] = f"{i}/{total} {name}"


def _ctx_banner(cmd, env_extra=None):

    e = env_extra or {}
    scene = e.get("DATASET", "")
    step = ""
    kv = []
    for i, a in enumerate(cmd):
        if a.endswith(".py"):
            step = os.path.basename(a)
        if a in (
            "--scene",
            "--dataset",
            "--split",
            "--pool",
            "--stage",
            "--variants",
            "--workers",
            "--model",
        ) and i + 1 < len(cmd):
            if a in ("--scene", "--dataset"):
                scene = cmd[i + 1]
            else:
                kv.append(f"{a.lstrip('-')}={cmd[i + 1]}")
    if not step and cmd:
        step = os.path.basename(cmd[0])
    arm = []
    if e.get("ABLATION_ARCH") and e["ABLATION_ARCH"] != "gru":
        arm.append(e["ABLATION_ARCH"])
    arm += [
        "no-" + k[len("ABLATION_USE_") :].lower()
        for k, v in e.items()
        if k.startswith("ABLATION_USE_")
        and v == "False"
        and k
        not in (
            "ABLATION_USE_HULL",
            "ABLATION_USE_WALK",
            "ABLATION_USE_SHAPE",
            "ABLATION_USE_ZOOMSHUF",
            "ABLATION_USE_VPRPATCHSHUF",
        )
    ]
    if e.get("ABLATION_USE_ZOOMSHUF") == "True":
        arm.append("zoomshuf")
    if e.get("ABLATION_USE_VPRPATCHSHUF") == "True":
        arm.append("vprpatchshuf")
    if e.get("DATA_ABLATION_RANDOM_WINDOW") == "True":
        arm.append("randwin")
    if e.get("STEP_MAX_DIST_TAG"):
        arm.append("stepmax-" + e["STEP_MAX_DIST_TAG"])
    bits = [f"场景={scene or '-'}", f"步骤={step}"]
    bits += kv
    if e.get("VPR_MODEL_TYPE") or e.get("MODEL"):
        bits.append(f"模型={e.get('VPR_MODEL_TYPE') or e.get('MODEL')}")
    if e.get("TRAIN_SEED"):
        bits.append(f"种子={e['TRAIN_SEED']}")
    bits.append(f"臂={','.join(arm) if arm else 'default'}")
    prog = " ".join(f"[{k} {v}]" for k, v in _PROG.items())
    return "▶ " + (prog + " " if prog else "") + " | ".join(bits)


def run(cmd, env_extra=None, required=True):

    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    prefix = " ".join(f"{k}={v}" for k, v in (env_extra or {}).items())
    for attempt in range(TRANSIENT_RETRIES + 1):
        if attempt:
            print(
                f"\n[retry {attempt}/{TRANSIENT_RETRIES}] 上一次是瞬时 GPU 故障,"
                f"{RETRY_SLEEP_S}s 后重跑同一条命令",
                flush=True,
            )
            _log(f"[retry {attempt}] transient GPU failure, re-running")
            time.sleep(RETRY_SLEEP_S)
        print(f"\n{_ctx_banner(cmd, env_extra)}", flush=True)
        _log(_ctx_banner(cmd, env_extra))
        print(f"$ {prefix + ' ' if prefix else ''}{' '.join(cmd)}", flush=True)
        _log(f"$ {prefix + ' ' if prefix else ''}{' '.join(cmd)}")
        proc = subprocess.Popen(
            cmd,
            cwd=REPO,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        bar = ""
        transient = False
        for chunk in proc.stdout:
            if "\r" in chunk:
                bar = chunk.rsplit("\r", 1)[-1]
                continue
            text = (bar + chunk).replace("\r", "\n")
            bar = ""
            sys.stdout.write(text)
            sys.stdout.flush()
            for line in text.splitlines():
                _log_line(line)
                if any(m in line for m in TRANSIENT_GPU_MARKERS):
                    transient = True
        rc = proc.wait()
        if rc == 0 or not transient:
            break
    if rc and required:
        raise subprocess.CalledProcessError(rc, cmd)
    return rc == 0


CHANNELS_ON = ()
ARCH = None
RANDOM_WINDOW = False
THRESHOLD = None
STEP_MAX_MODE = "fixed"
FORCE_PREPARE = False
LABEL_RATIO = float(PAPER_CONFIG["producers"]["label_ratio"])
PAPER_THRESHOLDS = tuple(float(value) for value in PAPER_CONFIG["thresholds_m"][1:])


def _traj_env(stem):
    env = {
        "DATASET": stem,
        "MODEL": MAIN_BACKBONE.lower(),
        "VPR_MODEL_TYPE": MAIN_BACKBONE,
        "ENABLE_SCENE_MODELS_CACHE": "True",
        "TRAIN_SEED": str(PAPER_SEEDS[0]),
        **FULL_VARIANT_ENV,
    }
    _apply_channels(env, CHANNELS_ON)
    if ARCH:
        env["ABLATION_ARCH"] = ARCH
    if RANDOM_WINDOW:
        env["DATA_ABLATION_RANDOM_WINDOW"] = "True"
    if THRESHOLD is not None:
        env["SUCCESS_THRESHOLD_TEST"] = f"{THRESHOLD:g}"
        env["SUCCESS_THRESHOLD_TRAIN"] = f"{THRESHOLD * LABEL_RATIO:g}"
    if STEP_MAX_MODE == "auto":
        env["STEP_MAX_DIST_TAG"] = "auto"
    return env


def _has_images(path):
    if not os.path.isdir(path):
        return False
    for _root, _dirs, files in os.walk(path):
        if any(name.lower().endswith(IMAGE_EXTENSIONS) for name in files):
            return True
    return False


def _prepare_portable_raw_input(stem, force=False):

    from core.config import resolve_dataset

    prepare = resolve_dataset(stem).get("prepare") or {}
    if not prepare:
        return
    tool = prepare.get("tool")
    if not tool:
        raise SystemExit(f"[{stem}] prepare block has no tool")
    tool_path = tool if os.path.isabs(tool) else os.path.join(REPO, tool)
    if not os.path.isfile(tool_path):
        raise SystemExit(f"[{stem}] prepare tool does not exist: {tool_path}")
    command = [PY, tool, "--dataset", stem]
    if force:
        command.append("--force")
    run(command)


def _require_portable_raw_input(stem):

    from core.config import resolve_dataset

    cfg = resolve_dataset(stem)
    build = cfg.get("build") or {}
    raw_rel = build.get("raw_root")
    if not raw_rel:
        raise SystemExit(f"[{stem}] dataset YAML has no build.raw_root")
    raw_root = raw_rel if os.path.isabs(raw_rel) else os.path.join(REPO, raw_rel)
    portable_root = os.path.realpath(
        os.path.join(REPO, "datasets", "database_raw_processed")
    )
    if os.path.commonpath([os.path.realpath(raw_root), portable_root]) != portable_root:
        raise SystemExit(
            f"[{stem}] raw_root is not portable: {raw_root}\n"
            "It must be inside datasets/database_raw_processed/."
        )
    if not os.path.isdir(raw_root):
        raise SystemExit(
            f"[{stem}] missing processed input: {raw_root}\n"
            "Copy the complete datasets/database_raw_processed/ tree to this device."
        )

    image_subdir = build.get("image_subdir", "undistorted_images")
    splits = build.get("splits") or {}
    required_recordings = sorted(
        {
            rec
            for split in ("database", "train", "val", "test")
            for rec in (splits.get(split) or [])
        }
    )
    missing_splits = []
    for split in ("database", "train", "val", "test"):
        recordings = splits.get(split) or []
        if not recordings:
            missing_splits.append(f"{split}: no recordings configured")
            continue
        usable = False
        for rec in recordings:
            rec_dir = os.path.join(raw_root, rec)
            image_dir = (
                rec_dir if image_subdir == "." else os.path.join(rec_dir, image_subdir)
            )
            if _has_images(image_dir):
                usable = True
                break
        if not usable:
            missing_splits.append(f"{split}: none of {recordings} contains images")
    if missing_splits:
        raise SystemExit(
            f"[{stem}] processed input is incomplete:\n  - "
            + "\n  - ".join(missing_splits)
        )

    links = []
    for dirpath, dirs, files in os.walk(raw_root):
        for name in dirs + files:
            path = os.path.join(dirpath, name)
            if os.path.islink(path):
                links.append(os.path.relpath(path, REPO))
                if len(links) == 5:
                    break
        if len(links) == 5:
            break
    if links:
        raise SystemExit(
            f"[{stem}] processed input is not self-contained; symlinks found:\n  - "
            + "\n  - ".join(links)
            + "\nMaterialize these files before copying to another device."
        )
    print(
        f"[input ok] {os.path.relpath(raw_root, REPO)} ({len(required_recordings)} recordings)"
    )


def _layout_ready(scene):
    base = os.path.join(REPO, "datasets", scene)
    return all(
        _has_images(os.path.join(base, part))
        for part in ("database", "queries_train", "queries_val", "queries_test")
    )


def _layout_params_changed(scene, stem):

    try:
        from tools.layout_manifest import generation_params, load_manifest
    except Exception:
        return ""
    old = (load_manifest(scene) or {}).get("params")
    if not old:
        return ""
    import json as _json
    import subprocess as _sp

    probe = (
        "import json;from tools.layout_manifest import generation_params;"
        "print('<<PARAMS>>' + json.dumps(generation_params()))"
    )
    try:
        out = _sp.run(
            [PY, "-c", probe],
            cwd=REPO,
            env={**os.environ, "DATASET": stem},
            capture_output=True,
            text=True,
            timeout=120,
        )
        marker = out.stdout.rsplit("<<PARAMS>>", 1)
        if out.returncode or len(marker) != 2:
            return ""
        now = _json.loads(marker[1].splitlines()[0])
    except Exception:
        return ""
    diffs = []
    for group in set(old) | set(now):
        a, b = old.get(group), now.get(group)
        if isinstance(a, dict) and isinstance(b, dict):
            for key in set(a) | set(b):
                if a.get(key) != b.get(key):
                    diffs.append(f"{group}.{key}: {a.get(key)} -> {b.get(key)}")
        elif a != b:
            diffs.append(f"{group}: {a} -> {b}")
    return "; ".join(sorted(diffs))


def _require_layout(scene, ran_layout=False):
    if not _layout_ready(scene):
        hint = (
            "the layout stage just ran and still left it incomplete -- 看它的输出"
            if ran_layout
            else f"先跑 `python reproduce.py --scene <数据集> --stage layout`"
            " —— --stage 一次只跑一个阶段,build 不会顺带建布局"
        )
        raise SystemExit(
            f"[{scene}] layout is incomplete: {hint}. Expected non-empty "
            f"database/ and queries_train|val|test/ under datasets/{scene}/"
        )
    _require_unique_basenames(scene)


def _require_unique_basenames(scene):

    import collections

    base = os.path.join(REPO, "datasets", scene)
    for part, key in (
        ("queries_train", "stem"),
        ("queries_val", "stem"),
        ("queries_test", "stem"),
        ("database", "name"),
    ):
        counts = collections.Counter()
        for dirpath, _, files in os.walk(os.path.join(base, part)):
            for name in files:
                if name.lower().endswith(IMAGE_EXTENSIONS):
                    counts[name.split(".")[0] if key == "stem" else name] += 1
        dupes = [s for s, n in counts.items() if n > 1]
        if dupes:
            raise SystemExit(
                f"[{scene}/{part}] {len(dupes)} of {len(counts)} filenames appear "
                f"in more than one recording (e.g. '{sorted(dupes)[0]}'), and a "
                f"frame is identified by its filename alone -- rows for different "
                f"frames would be merged, or a candidate would be replayed against "
                f"the wrong trajectory.\nGive the recordings disjoint filenames "
                f"(a multi-camera rig shares one clock: append the device id after "
                f"an '@', which the timestamp parser discards), or list only "
                f"recordings with disjoint filenames in this scene's YAML."
            )


def _require_sequence_assets(scene):
    dataset_root = os.path.join(REPO, "datasets", scene)
    database = os.path.join(dataset_root, "database")
    asset_root = os.path.join(REPO, "assets", "sequence_assets", scene)
    missing, present = [], []
    for dirpath, _dirs, files in os.walk(database):
        for name in files:
            if not name.lower().endswith(IMAGE_EXTENSIONS):
                continue
            image_path = os.path.join(dirpath, name)
            expected = os.path.join(
                asset_root, os.path.relpath(image_path, dataset_root) + ".pt"
            )
            if os.path.isfile(expected):
                present.append(expected)
            elif len(missing) < 5:
                missing.append(os.path.relpath(expected, REPO))
    if missing:
        raise SystemExit(
            f"[{scene}] offline sequence build finished with missing assets:\n  - "
            + "\n  - ".join(missing)
        )

    leftovers = [
        os.path.join(dirpath, name)
        for dirpath, _dirs, files in os.walk(asset_root)
        for name in files
        if ".pt.tmp." in name
    ]
    if leftovers:
        raise SystemExit(
            f"[{scene}] {len(leftovers)} interrupted sequence-asset writes are on "
            f"disk (e.g. {os.path.relpath(leftovers[0], REPO)}). Remove them and "
            "rerun the build stage."
        )


def _data_suffix(env):

    from core.config import data_tag

    return data_tag(
        random_window=env.get("DATA_ABLATION_RANDOM_WINDOW") == "True",
        step_max_tag=env.get("STEP_MAX_DIST_TAG", ""),
    )


def _configure_step_max(env, scene):

    if STEP_MAX_MODE != "auto":
        return
    path = os.path.join(REPO, "assets", "map_feature_rate", f"{scene}_walk.json")
    try:
        with open(path, encoding="utf-8") as handle:
            walk = json.load(handle)
        value = float(walk["step_max_dist_auto_p99"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SystemExit(
            f"[{scene}] auto step_max requested but verified walk has no proposal: {exc}"
        ) from exc
    if not math.isfinite(value) or value <= 0:
        raise SystemExit(f"[{scene}] invalid automatic step_max_dist: {value!r}")
    env["STEP_MAX_DIST_OVERRIDE"] = f"{value:.6f}"
    env["STEP_MAX_DIST_TAG"] = "auto"
    print(
        f"[step-max A/B] {scene}: mapping-only P99 = {value:.3f} m "
        "(isolated _stepmax-auto artifacts)"
    )


def _expected_steps(stem):

    from core.config import resolve_dataset

    seq = resolve_dataset(stem).get("sequence") or {}
    if not seq.get("spatial_steps"):
        raise SystemExit(
            f"[{stem}] configs/datasets/{stem}.yaml 的 sequence 段缺 spatial_steps "
            f"—— 没有默认值可退,场景必须自己声明"
        )
    return sum(seq["spatial_steps"]) + 1


def _require_state_cache(scene, split, suffix, stem, model_id=None):
    model_id = model_id or MODEL_ID
    expected = _expected_steps(stem)
    path = os.path.join(
        REPO,
        "assets",
        model_id,
        f"{split}_trajectories_{scene}{suffix}.json",
    )
    meta = path + ".meta.json"
    if not os.path.isfile(path) or os.path.getsize(path) <= 2:
        raise SystemExit(f"[{scene}/{split}] missing or empty state cache: {path}")
    if not os.path.isfile(meta):
        raise SystemExit(f"[{scene}/{split}] cache has no fingerprint sidecar: {meta}")
    try:
        with open(meta, "r", encoding="utf-8") as handle:
            stamp = json.load(handle)
            settings = stamp.get("settings") or {}
    except (OSError, ValueError) as exc:
        raise SystemExit(f"[{scene}/{split}] invalid cache sidecar: {meta}: {exc}")
    if not stamp.get("complete") or not (
        stamp.get("cache_key") or stamp.get("semantic_digest")
    ):
        raise SystemExit(
            f"[{scene}/{split}] state cache has an incomplete content key: {meta}"
        )
    if (
        settings.get("temporal_protocol") != 2
        or settings.get("expected_steps") != expected
    ):
        raise SystemExit(
            f"[{scene}/{split}] state cache is not verified as protocol-v2/"
            f"{expected}-step (sidecar says {settings.get('expected_steps')}): {meta}"
        )


def reproduce_trajectory(dataset, stages, force, seeds=None):
    seeds = tuple(seeds or (PAPER_SEEDS[0],))
    for stem in TRAJECTORY_DATASETS[dataset]:
        scene = SCENE_NAME[stem]
        env = _traj_env(stem)
        print(f"\n{'='*60}\n# {dataset}  partition: {stem}  (Scene={scene})\n{'='*60}")

        if "layout" in stages:
            _prepare_portable_raw_input(stem, force=FORCE_PREPARE)
            _require_portable_raw_input(stem)
            from core.config import resolve_dataset

            if resolve_dataset(stem).get("clean"):
                run([PY, "tools/clean_data.py", "--dataset", stem])
            else:
                print(f"[skip clean] {scene}: raw_root is already a final view")
            stale_params = _layout_params_changed(scene, stem)
            rebuild_layout = force or not _layout_ready(scene) or bool(stale_params)
            if stale_params:
                print(
                    f"[rebuild layout] {scene}: 建图参数与 .layout_manifest.json "
                    f"不一致 -> {stale_params}"
                )
            elif not rebuild_layout:
                print(f"[skip layout] datasets/{scene} exists; presence cache hit")
            if rebuild_layout:
                print(f"[rebuild layout] {scene}: regenerating from raw inputs")
                run([PY, "tools/build_paper_dataset.py", "--dataset", stem])
                run([PY, "tools/layout_manifest.py", "--dataset", stem, "--write"])
            _require_layout(scene, ran_layout=True)

        if "build" in stages:
            _require_layout(scene)
            sequence_env = dict(env)
            if force:
                sequence_env["FORCE_REBUILD_SEQUENCE_ASSETS"] = "1"
            run([PY, "tools/build_offline_sequences.py"], sequence_env)
            _require_sequence_assets(scene)
            _require_walk_table(scene, stem, env)
            _configure_step_max(env, scene)

            _splits = PAPER_CONFIG["producers"]["state_splits"]
            if BUILD_SPLITS:
                _splits = [s for s in _splits if s in BUILD_SPLITS]
            for split in _splits:
                workers = os.getenv(
                    "BUILD_WORKERS",
                    str(PAPER_CONFIG["producers"]["default_build_workers"]),
                )
                if workers == "1":
                    run(
                        [PY, f"sl_agent/build_sl_dataset_{split}.py"],
                        dict(env, FORCE_REBUILD_CACHE="1") if force else env,
                    )
                else:
                    cmd = [
                        PY,
                        "tools/parallel_build_sl.py",
                        "--dataset",
                        stem,
                        "--split",
                        split,
                        "--workers",
                        workers,
                    ]
                    if force:
                        cmd.append("--force")
                    run(cmd, env)
                _require_state_cache(scene, split, _data_suffix(env), stem=stem)

            for tool in PAPER_CONFIG["producers"]["injection_tools"]:
                run(
                    [PY, f"tools/{tool}.py", "--scene", scene, "--model-id", MODEL_ID],
                    env,
                )

        if "train" in stages:
            env_train = dict(env)
            env_train["FORCE_RETRAIN"] = "True" if force else "False"
            for _si, seed in enumerate(seeds, 1):
                _prog("种子", _si, len(seeds), str(seed))
                run(
                    [
                        PY,
                        os.path.join(
                            os.path.dirname(os.path.abspath(__file__)), "train_agent.py"
                        ),
                    ],
                    dict(env_train, TRAIN_SEED=str(seed)),
                )
            _PROG.pop("种子", None)

        if "eval" in stages:
            scored = []
            for seed in seeds:
                mark = _run_log_size()
                run([PY, "tools/fast_offline_eval.py"], dict(env, TRAIN_SEED=str(seed)))
                pct = _last_ours_pct(mark)
                if pct is not None:
                    scored.append((seed, pct))
            vals = [p for _, p in scored]
            if vals:
                mean = sum(vals) / len(vals)
                std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
                detail = ", ".join(f"{s}:{p:.2f}" for s, p in scored)
                line = (
                    f"[seed-mean] {scene}: {mean:.2f}+-{std:.2f} over "
                    f"{len(vals)}/{len(seeds)} seeds ({detail})"
                )
            else:
                line = (
                    f"[seed-mean] {scene}: no 'Ours' row parsed from "
                    f"{len(seeds)} evals"
                )
            print(line, flush=True)
            _log(line)


def _require_walk_table(scene, stem, env):

    from core.config import resolve_dataset

    sequence = resolve_dataset(stem).get("sequence") or {}
    if not sequence.get("step_feature_tau"):
        return
    if sum(sequence.get("spatial_steps") or [0, 0]) == 0:
        print(f"[skip walk] {scene}: spatial_steps [0, 0], flat database")
        return
    path = os.path.join(REPO, "assets", "map_feature_rate", f"{scene}_walk.json")
    print(f"\n# walk provenance: {scene}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        if STEP_MAX_MODE != "auto":
            print(f"[skip walk] {scene}: existing preprocessing table")
            return
        try:
            with open(path, encoding="utf-8") as handle:
                if json.load(handle).get("step_max_dist_auto_p99") is not None:
                    print(f"[skip walk] {scene}: existing table includes auto proposal")
                    return
        except (OSError, ValueError):
            pass
        print(f"[walk] {scene}: existing table lacks auto proposal; re-emitting")

    for k in (1, 3, 6, 12, 24):
        stride_path = os.path.join(
            REPO, "assets", "map_feature_rate", f"{scene}_s{k}.json"
        )
        if os.path.isfile(stride_path) and os.path.getsize(stride_path) > 0:
            print(f"[skip stride] {scene}/{k}: existing preprocessing table")
            continue
        print(f"#   stride {k}: missing, building")
        run(
            [PY, "tools/map_feature_rate.py", "--scene", scene, "--stride", str(k)], env
        )
    run([PY, "tools/map_feature_rate.py", "--scene", scene, "--emit-walk"], env)
    if not (os.path.isfile(path) and os.path.getsize(path) > 0):
        raise SystemExit(f"[{scene}] walk 表没能生成: {path}")
    _write_calibrated_budget(scene, stem, path)


def _write_calibrated_budget(scene, stem, walk_path):

    import yaml as _yaml

    cfg_path = os.path.join(REPO, "configs", "datasets", f"{stem}.yaml")
    with open(cfg_path, encoding="utf-8") as fh:
        body = fh.read()
    cfg = _yaml.safe_load(body) or {}
    seq = cfg.get("sequence") or {}
    if "step_budget_calibrated" not in seq:
        return
    with open(walk_path, encoding="utf-8") as fh:
        w = _yaml.safe_load(fh)
    budget = w.get("budget_half")
    if budget is None:
        raise SystemExit(f"[{scene}] {walk_path} 里没有 budget_half,无法标定")
    if (
        seq.get("step_budget_calibrated") is True
        and seq.get("step_budget") is not None
        and abs(float(seq["step_budget"]) - float(budget)) <= 1e-12
    ):
        return
    if "sequence:" not in body:
        raise SystemExit(
            f"[{scene}] {cfg_path} 没有 sequence 段 —— 先补上 spatial_steps / "
            f"step_feature_tau / step_max_dist,再跑一次(budget 会自动填)"
        )
    if seq.get("step_budget") is None:
        body = body.replace(
            "sequence:\n",
            f"sequence:\n"
            f"  # step_budget 由 tools/map_feature_rate.py 标定后自动写入,不是人选的:\n"
            f"  # 在 stride 1/3/6/12/24 上量直接视野变化,取该图自身饱和值的一个固定比例。\n"
            f"  # 本图:饱和 {w.get('saturation')},单步率 {w.get('rate')},"
            f"链式底噪 {w.get('offset')}\n"
            f"  step_budget: {budget}\n",
            1,
        )
    else:
        body = re.sub(
            r"(?m)^  step_budget:.*$", f"  step_budget: {budget}", body, count=1
        )
    body = re.sub(
        r"(?m)^  step_budget_calibrated:.*$",
        "  step_budget_calibrated: true",
        body,
        count=1,
    )
    temporary = f"{cfg_path}.tmp.{os.getpid()}"
    try:
        with open(temporary, "w", encoding="utf-8") as fh:
            fh.write(body)
        os.replace(temporary, cfg_path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    print(f"# step_budget: {scene} 标定为 {budget:.6f},已写入 {cfg_path}")


def _paper_variants():

    return [
        (row["label"], dict(row["variant"]))
        for row in PAPER_CONFIG["ablation_rows"]
        if row.get("variant") is not None
        and not row.get("default")
        and not row.get("supplementary")
    ]


def _train_workers():

    configured = int(PAPER_CONFIG["producers"].get("train_workers", 1))
    try:
        with open("/proc/meminfo") as handle:
            available_gb = next(
                int(line.split()[1]) / 1048576
                for line in handle
                if line.startswith("MemAvailable:")
            )
    except (OSError, StopIteration):
        return 1
    fits = max(1, int(available_gb // 3))
    workers = max(1, min(configured, fits))
    if workers < configured:
        print(
            f"[train] 并行度由 {configured} 降到 {workers}"
            f"(可用内存 {available_gb:.1f} GB,每路留 3 GB)",
            flush=True,
        )
    return workers


def _train_one(stem, label, variant, seed, force):

    env = os.environ.copy()
    env.update(_traj_env(stem))
    env.update(variant)
    env["TRAIN_SEED"] = str(seed)
    env["FORCE_RETRAIN"] = "True" if force else "False"
    env_extra = {**_traj_env(stem), **variant, "TRAIN_SEED": str(seed)}
    prefix = " ".join(f"{k}={v}" for k, v in env_extra.items())
    scene = SCENE_NAME[stem]
    head = [
        f"\n# paper weight: {scene} / {label} / seed={seed}",
        _ctx_banner([PY, "train_agent.py"], env_extra),
        f"$ {prefix} {PY} train_agent.py",
    ]
    for attempt in range(TRANSIENT_RETRIES + 1):
        if attempt:
            head.append(
                f"[retry {attempt}/{TRANSIENT_RETRIES}] 上一次是瞬时 GPU 故障,"
                f"{RETRY_SLEEP_S}s 后重跑同一条命令"
            )
            time.sleep(RETRY_SLEEP_S)
        proc = subprocess.run(
            [PY, "train_agent.py"],
            cwd=REPO,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        out = proc.stdout or ""
        if proc.returncode == 0 or not any(m in out for m in TRANSIENT_GPU_MARKERS):
            break
    body = [
        line
        for line in (proc.stdout or "").replace("\r", "\n").splitlines()
        if line.strip()
    ]
    return proc.returncode == 0, head + body[-12:]


def _train_seed_set(stems, variants, force, failures, seeds=PAPER_SEEDS):

    jobs = [
        (stem, label, variant, seed)
        for stem in stems
        for label, variant in variants
        for seed in seeds
    ]
    if not jobs:
        return
    workers = _train_workers()
    if workers == 1:
        for stem, label, variant, seed in jobs:
            ok, lines = _train_one(stem, label, variant, seed, force)
            _flush_train_output(lines)
            if not ok:
                failures.append(f"weight {SCENE_NAME[stem]}/{label}/seed={seed}")
        return
    print(f"\n[train] {len(jobs)} 个权重,{workers} 路并行", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_train_one, stem, label, variant, seed, force)
            for stem, label, variant, seed in jobs
        ]
        for (stem, label, variant, seed), future in zip(jobs, futures):
            try:
                ok, lines = future.result()
            except Exception as exc:
                ok, lines = False, [f"# 训练异常 {SCENE_NAME[stem]}/{label}: {exc}"]
            _flush_train_output(lines)
            if not ok:
                failures.append(f"weight {SCENE_NAME[stem]}/{label}/seed={seed}")


def _flush_train_output(lines):

    for line in lines:
        print(line, flush=True)
        _log_line(line)


def _build_backbone_test_caches(stems, force, failures):

    worker_policy = PAPER_CONFIG["producers"]["table2"]
    heavy = set(worker_policy["heavy_backbones"])
    worker_overrides = {
        name: int(value)
        for name, value in worker_policy.get("worker_overrides", {}).items()
    }
    env_workers = os.getenv("BUILD_WORKERS")
    default_splits = ",".join(worker_policy["default_splits"])
    splits = tuple(
        s.strip()
        for s in os.getenv("TABLE2_SPLITS", default_splits).split(",")
        if s.strip()
    )
    print(
        f"\n[cross-retriever] splits built this run: {', '.join(splits)}"
        + (
            ""
            if len(splits) > 1
            else "  (train/val 押后,用 TABLE2_SPLITS=test,train,val 补)"
        )
    )
    todo = [
        (sp, b, s)
        for sp in splits
        for s in stems
        for b in PAPER_BACKBONES
        if b != MAIN_BACKBONE
    ]
    for split, backbone, stem in todo:
        scene = SCENE_NAME[stem]
        model_id = _MODEL_ID_OF[backbone]
        env = _traj_env(stem)
        env.update(MODEL=backbone, VPR_MODEL_TYPE=backbone)
        workers = env_workers or str(
            worker_overrides.get(
                backbone,
                (
                    worker_policy["heavy_workers"]
                    if backbone in heavy
                    else worker_policy["default_workers"]
                ),
            )
        )
        cmd = [
            PY,
            "tools/parallel_build_sl.py",
            "--dataset",
            stem,
            "--split",
            split,
            "--workers",
            workers,
            "--model",
            backbone,
        ]
        if force:
            cmd.append("--force")
        print(f"\n# cross-retriever cache: {scene} / {backbone} / {split}")
        if not run(cmd, env, required=False):
            failures.append(f"cross-retriever cache {scene}/{backbone}/{split}")
            continue
        for tool in PAPER_CONFIG["producers"]["injection_tools"]:
            if not run(
                [
                    PY,
                    f"tools/{tool}.py",
                    "--scene",
                    scene,
                    "--model-id",
                    model_id,
                    "--splits",
                    split,
                ],
                env,
                required=False,
            ):
                failures.append(f"cross-retriever {tool} {scene}/{backbone}/{split}")
                break


def _prepare_paper_model_inputs(failures):

    cmd = [PY, "tools/prepare_model_checkpoints.py"]
    source = os.getenv("STAR_VPR_PRETRAINED_SOURCE")
    if source:
        cmd.extend(["--source-root", source])
    if not run(cmd, required=False):
        failures.append("public Table-II checkpoint preparation")


def _report_radii_args(stem):

    import yaml

    path = os.path.join(_ROOT, "configs", "datasets", f"{stem}.yaml")
    with open(path, encoding="utf-8") as handle:
        own = float(yaml.safe_load(handle)["thresholds"]["test_m"])
    if abs(own - float(PAPER_CONFIG["thresholds_m"][0])) < 1e-9:
        return []
    radii = sorted({own, *PAPER_THRESHOLDS})
    return ["--success-thresholds", ",".join(f"{r:g}" for r in radii)]


def _build_table1_baselines(stems, failures, force=False):

    for stem in stems:
        scene = SCENE_NAME[stem]
        env = _traj_env(stem)
        for label, producer in PAPER_CONFIG["classic_baselines"].items():
            if producer.get("phase") != "table1":
                continue
            print(f"\n# baseline: {scene} / {label}")
            if not run(
                [PY, producer["script"], producer["scene_flag"], scene],
                env,
                required=False,
            ):
                failures.append(f"baseline {scene}/{label}")

        for label, producer in PAPER_CONFIG["learned_baselines"].items():
            weight_tiers = (
                (float(PAPER_CONFIG["thresholds_m"][0]),) + PAPER_THRESHOLDS
                if producer.get("per_threshold")
                else (None,)
            )
            for pool in PAPER_CONFIG["reranker_pools"]:
                for th in weight_tiers:
                    tag = label if th is None else f"{label}@th{th:g}"
                    print(f"\n# baseline: {scene} / {tag} / pool {pool}")
                    command = [
                        PY,
                        producer["script"],
                        "--scenes",
                        scene,
                        "--pool",
                        pool,
                    ]
                    if th is not None:
                        command += ["--threshold", f"{th:g}"]
                    else:
                        command += _report_radii_args(stem)
                    command.extend(producer.get("extra_args", []))
                    if force:
                        command.append("--force")
                    if not run(command, env, required=False):
                        failures.append(f"baseline {scene}/{tag}/{pool}")


def _build_random_window_caches(stems, force, failures):

    random_window = PAPER_CONFIG["producers"]["random_window"]
    workers = os.getenv("BUILD_WORKERS", str(random_window["workers"]))
    suffix = _data_suffix({"DATA_ABLATION_RANDOM_WINDOW": "True"})
    for stem in stems:
        scene = SCENE_NAME[stem]
        env = _traj_env(stem)
        env["DATA_ABLATION_RANDOM_WINDOW"] = "True"
        for split in random_window["splits"]:
            cmd = [
                PY,
                "tools/parallel_build_sl.py",
                "--dataset",
                stem,
                "--split",
                split,
                "--workers",
                workers,
                "--model",
                MAIN_BACKBONE,
            ]
            if force:
                cmd.append("--force")
            print(f"\n# random-window cache: {scene} / {split}")
            if not run(cmd, env, required=False):
                failures.append(f"random-window cache {scene}/{split}")
                continue
            try:
                _require_state_cache(scene, split, suffix, stem)
            except SystemExit as exc:
                print(exc, file=sys.stderr)
                failures.append(f"random-window cache {scene}/{split}")
        rw_env = dict(env, DATA_ABLATION_RANDOM_WINDOW="True")
        for tool in PAPER_CONFIG["producers"]["injection_tools"]:
            if not run(
                [PY, f"tools/{tool}.py", "--scene", scene, "--model-id", MODEL_ID],
                rw_env,
                required=False,
            ):
                failures.append(f"random-window {tool} {scene}")


def _run_table_replay(dataset=None, tables=None, datasets=None):

    cmd = [PY, "run_paper_tables.py"]
    if tables:
        cmd.extend(["--tables", tables])
    if dataset is not None:
        cmd.extend(["--datasets", dataset])
        if tables is None and dataset not in TABLE3_DATASETS:
            cmd.extend(["--tables", "main,backbones"])
    elif datasets:
        cmd.extend(["--datasets", ",".join(datasets)])
    reference_stem = next(iter(PAPER_CONFIG["scene_stems"].values()))
    env = dict(
        FULL_VARIANT_ENV,
        DATASET=reference_stem,
        MODEL=MAIN_BACKBONE,
        VPR_MODEL_TYPE=MAIN_BACKBONE,
        ENABLE_SCENE_MODELS_CACHE="True",
    )
    run(cmd, env, required=False)


def phase_table3_draft(dataset, force, failures):
    phase_table3(dataset, force, failures, draft=True)


def phase_table1_draft(dataset, force, failures):

    stems = TRAJECTORY_DATASETS[dataset]
    reproduce_trajectory(dataset, ["layout", "build", "train", "eval"], force)
    _train_seed_set(
        stems,
        [("Default Full Model", FULL_VARIANT_ENV)],
        force,
        failures,
        seeds=PAPER_SEEDS[:1],
    )
    _prepare_paper_model_inputs(failures)
    _build_table1_baselines(stems, failures, force)


def _train_at_thresholds(dataset, force, failures, variants, seeds):

    global THRESHOLD
    stems = TRAJECTORY_DATASETS[dataset]
    prev = THRESHOLD
    try:
        for th in PAPER_THRESHOLDS:
            THRESHOLD = th
            print(
                f"\n[threshold] {dataset}: 重训 @ {th:g} m"
                f"(标签 {th * LABEL_RATIO:g} m,seeds={list(seeds)})"
            )
            _train_seed_set(stems, variants, force, failures, seeds=seeds)
    finally:
        THRESHOLD = prev


def phase_table1_thresholds(dataset, force, failures):

    _train_at_thresholds(
        dataset,
        force,
        failures,
        [("Default Full Model", FULL_VARIANT_ENV)],
        seeds=PAPER_SEEDS[:1],
    )


def phase_table1_seeds(dataset, force, failures):

    stems = TRAJECTORY_DATASETS[dataset]
    _train_seed_set(
        stems,
        [("Default Full Model", FULL_VARIANT_ENV)],
        force,
        failures,
        seeds=PAPER_SEEDS[1:],
    )
    _train_at_thresholds(
        dataset,
        force,
        failures,
        [("Default Full Model", FULL_VARIANT_ENV)],
        seeds=PAPER_SEEDS[1:],
    )


def phase_table3_thresholds(dataset, force, failures):

    if dataset not in TABLE3_DATASETS:
        return
    variants = [
        (row["label"], row["variant"])
        for row in PAPER_CONFIG["ablation_rows"]
        if row.get("variant") is not None
        and not row.get("default")
        and not row.get("supplementary")
        and row.get("threshold_sweep", True)
    ]
    _train_at_thresholds(dataset, force, failures, variants, seeds=PAPER_SEEDS)


def phase_table2(dataset, force, failures):

    os.environ["TABLE2_SPLITS"] = ",".join(
        PAPER_CONFIG["producers"]["table2"]["default_splits"]
    )
    _build_backbone_test_caches(TRAJECTORY_DATASETS[dataset], force, failures)


def phase_table3(dataset, force, failures, draft=False):

    if dataset not in TABLE3_DATASETS:
        return
    stems = TRAJECTORY_DATASETS[dataset]
    variants = _paper_variants()
    if draft:
        variants = [
            (row["label"], row["variant"])
            for row in PAPER_CONFIG["ablation_rows"]
            if row.get("variant") is not None
            and not row.get("default")
            and not row.get("supplementary")
            and row.get("draft", True)
        ]
        _train_seed_set(
            stems,
            variants,
            force,
            failures,
            seeds=(PAPER_SEEDS[0],),
        )
        return
    if any(
        (row.get("overrides") or {}).get("DATA_ABLATION_RANDOM_WINDOW")
        for row in PAPER_CONFIG["ablation_rows"]
    ):
        _build_random_window_caches(stems, force, failures)
    if any(
        (row.get("overrides") or {}).get("ABLATION_USE_ZOOMSHUF")
        for row in PAPER_CONFIG["ablation_rows"]
    ):
        for stem in stems:
            scene = SCENE_NAME[stem]
            run(
                [
                    PY,
                    "tools/inject_zoom_channel.py",
                    "--scene",
                    scene,
                    "--model-id",
                    MODEL_ID,
                    "--shuffle",
                    str(PAPER_CONFIG["producers"]["zoom_shuffle_seed"]),
                ],
                _traj_env(stem),
                required=False,
            )
    _train_seed_set(stems, _paper_variants(), force, failures)


def phase_table2_trainval(dataset, force, failures):

    stems = TRAJECTORY_DATASETS[dataset]
    os.environ["TABLE2_SPLITS"] = "train,val"
    try:
        _build_backbone_test_caches(stems, force, failures)
    finally:
        os.environ.pop("TABLE2_SPLITS", None)

    for stem in stems:
        scene = SCENE_NAME[stem]
        for _bi, backbone in enumerate(PAPER_BACKBONES, 1):
            _prog("骨干", _bi, len(PAPER_BACKBONES), backbone)
            if backbone == MAIN_BACKBONE:
                continue
            for threshold in (None,) + PAPER_THRESHOLDS:
                env = _traj_env(stem)
                env.update(MODEL=backbone, VPR_MODEL_TYPE=backbone)
                if threshold is not None:
                    env["SUCCESS_THRESHOLD_TEST"] = f"{threshold:g}"
                    env["SUCCESS_THRESHOLD_TRAIN"] = f"{threshold * LABEL_RATIO:g}"
                for seed in PAPER_SEEDS:
                    train_env = dict(env, TRAIN_SEED=str(seed))
                    if force:
                        train_env["FORCE_RETRAIN"] = "True"
                    label = f"{threshold:g}m" if threshold is not None else "5m"
                    print(
                        f"\n# retrained head: {scene} / {backbone} / {label} / seed {seed}"
                    )
                    if not run([PY, "train_agent.py"], train_env, required=False):
                        failures.append(
                            f"retrained head {scene}/{backbone}/{label}/seed{seed}"
                        )

    _run_table_replay(dataset, tables="backbones_retrained")


def _paper_completion_phases(include_table2_retrained=False):

    phases = [
        ("main table: remaining seeds, 10/15 m", phase_table1_seeds, "table1-seeds"),
        ("ablation arms at 10/15 m", phase_table3_thresholds, "table3-th"),
        ("ablation: all seeds", phase_table3, "table3"),
    ]
    if include_table2_retrained:
        phases.append(
            ("cross-retriever train/val", phase_table2_trainval, "table2-trainval")
        )
    return phases


def _union_head_jobs():

    head = PAPER_CONFIG["default_head"]
    members = list(head["members"])
    spec = PAPER_CONFIG.get("union_heads") or {}
    jobs = [("default", head["scene"], members, None, {})]
    for radius in spec.get("radii") or ():
        jobs.append(
            (f"radius {float(radius):g} m", head["scene"], members, float(radius), {})
        )
    for row in PAPER_CONFIG["ablation_rows"]:
        if row.get("default") or row.get("variant") is None:
            continue
        overrides = {
            key: value
            for key, value in row["variant"].items()
            if FULL_VARIANT_ENV.get(key) != value
        }
        if overrides:
            jobs.append((row["label"], head["scene"], members, None, overrides))
    if spec.get("leave_one_out"):
        for dataset, ds_spec in PAPER_CONFIG["datasets"].items():
            held = set(ds_spec["scenes"])
            if not held <= set(members):
                continue
            kept = [scene for scene in members if scene not in held]
            jobs.append((f"leave out {dataset}", f"LOO_{dataset}", kept, None, {}))
    return jobs


def _train_union_head(label, tag, scenes, radius, overrides, force, failures):
    cmd = [
        PY,
        "tools/train_union_head.py",
        "--scenes",
        ",".join(scenes),
        "--tag",
        tag,
        "--seeds",
        ",".join(str(seed) for seed in PAPER_SEEDS),
    ]
    if radius is not None:
        cmd += ["--threshold", f"{radius:g}"]
    for key, value in overrides.items():
        cmd += ["--set", f"{key}={value}"]
    if force:
        cmd.append("--force")
    print(f"\n# union head: {tag} / {label}", flush=True)
    if not run(cmd, required=False):
        failures.append(f"union head {tag}/{label}")


def phase_union_heads(force, failures):

    jobs = _union_head_jobs()
    for index, (label, tag, scenes, radius, overrides) in enumerate(jobs, 1):
        _prog("通用头", index, len(jobs), f"{tag} / {label}")
        _train_union_head(label, tag, scenes, radius, overrides, force, failures)


def phase_union_mix(force, failures):

    del force
    if not run([PY, "tools/run_union_mix.py"], required=False):
        failures.append(
            f"union head {(PAPER_CONFIG.get('union_heads') or {}).get('mix_tag')}"
        )


def _resolve_paper_order(arguments):

    if not arguments:
        return list(PAPER_DATASETS)
    order = [
        item.strip()
        for argument in arguments
        for item in argument.split(",")
        if item.strip()
    ]
    unknown = [item for item in order if item not in PAPER_DATASETS]
    duplicates = sorted({item for item in order if order.count(item) > 1})
    if not order or unknown or duplicates:
        details = []
        if unknown:
            details.append("unknown: " + ",".join(unknown))
        if duplicates:
            details.append("duplicated: " + ",".join(duplicates))
        suffix = f" ({'; '.join(details)})" if details else ""
        raise SystemExit(
            "--paper-order must list a non-empty, unique subset of: "
            + ",".join(PAPER_DATASETS)
            + suffix
        )
    return order


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--scene",
        required=False,
        choices=list(TRAJECTORY_DATASETS),
        help="isolated low-level trajectory target. Paper reproduction should "
        "use --paper-order, including for a single dataset.",
    )
    ap.add_argument(
        "--paper-order",
        nargs="+",
        metavar="DATASET",
        help="select a non-empty paper-dataset subset and set its order within "
        "every easy-first phase, e.g. --paper-order "
        "Boreas CustomCampus NCLT 4Seasons skips RobotCar. This never "
        "changes the phase order.",
    )
    ap.add_argument(
        "--stage", default="all", choices=["layout", "build", "train", "eval", "all"]
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="rebuild cached artifacts instead of skipping them",
    )
    ap.add_argument(
        "--force-prepare",
        action="store_true",
        help="reconvert dataset-specific raw input. Separate from --force so a "
        "state/layout rebuild never re-encodes tens of thousands of RobotCar frames.",
    )
    ap.add_argument(
        "--step-max-mode",
        choices=["fixed", "auto"],
        default="fixed",
        help=argparse.SUPPRESS,
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="change the SUCCESS THRESHOLD -- the radius within which a "
        "retrieval counts as correct (not an ROC operating point). "
        "Evaluates at this radius AND "
        "retrain the self-supervised labels around "
        f"{LABEL_RATIO:g}x it (configured paper label ratio). The threshold enters "
        "agent_tag, so each threshold gets its own weights instead of "
        "overwriting the scene's published ones. Omit to use the scene YAML.",
    )
    ap.add_argument(
        "--channels",
        default="",
        help=argparse.SUPPRESS,
    )
    ap.add_argument(
        "--arch",
        default=None,
        choices=[
            "gru",
            "mlp",
            "transformer",
            "single_frame",
            "gru_single",
            "mlp_flat",
            "mlp_pool",
            "single_frame_linear",
        ],
        help=argparse.SUPPRESS,
    )
    ap.add_argument(
        "--random-window",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    ap.add_argument(
        "--only",
        help="只跑点名的阶段(逗号分隔),其余跳过。可选:table1, table1-th, table2, "
        "table3-draft, table1-seeds, table3-th, table3, union, union-mix。补一格数据时用它,"
        "免得把八个阶段重走一遍 —— 即便已有资产会跳过,光走流程也要好几分钟才"
        "轮到目标阶段。仅对 --paper-order 生效",
    )
    ap.add_argument(
        "--table2-retrained",
        dest="table2_retrained",
        action="store_true",
        default=bool(PAPER_CONFIG["producers"]["table2"]["build_retrained_default"]),
        help=argparse.SUPPRESS,
    )
    ap.add_argument(
        "--no-table2-retrained",
        dest="table2_retrained",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    ap.add_argument(
        "--seeds",
        default=str(PAPER_SEEDS[0]),
        help="comma-separated TRAIN_SEED values for the train/eval stages of a "
        "--scene run (default: the frozen seed only, i.e. unchanged "
        "behaviour). With more than one, every seed is trained and "
        "evaluated and the mean+-std is printed. Paper datasets always "
        "use the six frozen seeds regardless of this option.",
    )
    ap.add_argument(
        "--crop-sweep",
        action="store_true",
        help="视野鲁棒性表:派生裁剪档场景、训练 mix 混采权重、与最强基线在同池"
        "同 test 上逐档对比,出 results/tables/crop_sweep.csv,然后退出。",
    )
    ap.add_argument(
        "--crop-variants",
        default=None,
        help="with --crop-sweep: 参与出表的裁剪档,默认 a4050,a5060,a7080,full",
    )
    ap.add_argument(
        "--crop-baselines",
        default=None,
        help="with --crop-sweep: 参与对比的学习型重排器,默认 FoL,PairVPR-s2",
    )
    ap.add_argument(
        "--splits",
        default=None,
        help="只建这几个 split 的状态缓存(逗号分隔,如 test)。换基座骨干做"
        "零样本迁移评测时只需要 test —— train/val 是最贵的两份,建了也用不上。",
    )
    ap.add_argument(
        "--model",
        default=None,
        help="换基座检索骨干(默认走论文协议的 MegaLoc)。**必须用这个参数** —— "
        "环境变量 MODEL/VPR_MODEL_TYPE 会被 core/env_isolation 剥掉(防止上一个 "
        "shell 泄漏进复现),给本入口传环境变量是无声无效的。",
    )
    ap.add_argument(
        "--crop-seeds",
        default="",
        help="逗号分隔的种子列表,默认用 run_crop_sweep 的六个种子",
    )
    ap.add_argument(
        "--crop-pool",
        default=None,
        help="with --crop-sweep: 候选池,默认 MegaLoc_8448",
    )
    ap.add_argument(
        "--crop-skip-baselines",
        action="store_true",
        help="with --crop-sweep: 只出 Raw 与本方法两行(快速自查)",
    )
    ap.add_argument(
        "--crop-heads",
        default=None,
        help="with --crop-sweep: '行名=权重场景;...' 评测已训好的头(如 UnionMaps / "
        "UnionMapsMix),不训单图 mix 头",
    )
    ap.add_argument(
        "--verify",
        action="store_true",
        help="read-only health check of all three data layers, then exit. Builds "
        "nothing. Run it on both machines and diff with "
        "tools/compare_machines.py.",
    )
    ap.add_argument(
        "--deep",
        action="store_true",
        help="with --verify: re-hash layer ② contents instead of comparing sizes",
    )
    ap.add_argument(
        "--json",
        dest="json_path",
        help="with --verify: write a machine-readable report to this path",
    )
    a = ap.parse_args()

    if a.scene and a.paper_order:
        ap.error("--scene and --paper-order are mutually exclusive")

    global CHANNELS_ON, THRESHOLD, ARCH, RANDOM_WINDOW, STEP_MAX_MODE, FORCE_PREPARE
    global MAIN_BACKBONE, MODEL_ID, BUILD_SPLITS
    if a.splits:
        BUILD_SPLITS = tuple(x.strip() for x in a.splits.split(",") if x.strip())
        bad = [x for x in BUILD_SPLITS if x not in ("train", "val", "test")]
        if bad:
            raise SystemExit(f"--splits 只能是 train/val/test,收到 {bad}")
        print(f"[splits] 只建 {', '.join(BUILD_SPLITS)} 的状态缓存")
    if a.model:
        match = [b for b in _MODEL_ID_OF if b.lower() == a.model.lower()]
        if not match:
            raise SystemExit(
                f"--model {a.model!r} 不认识;可选:{', '.join(sorted(_MODEL_ID_OF))}"
            )
        MAIN_BACKBONE = match[0]
        MODEL_ID = _MODEL_ID_OF[MAIN_BACKBONE]
        print(f"[model] 基座骨干改为 {MAIN_BACKBONE}(池 {MODEL_ID})")
    THRESHOLD = a.threshold
    STEP_MAX_MODE = a.step_max_mode
    FORCE_PREPARE = a.force_prepare
    ARCH = a.arch
    RANDOM_WINDOW = a.random_window
    CHANNELS_ON = tuple(c.strip() for c in a.channels.split(",") if c.strip())
    bad = [
        c
        for c in CHANNELS_ON
        if not (
            c in CHANNEL_GROUPS
            or (
                c.startswith("no-")
                and (c[3:] in BASE_GROUPS or c[3:] in CHANNEL_GROUPS)
            )
        )
    ]
    if bad:
        raise SystemExit(
            f"unknown --channels: {bad}\n"
            f"  打开:{sorted(CHANNEL_GROUPS)}\n"
            f"  关掉:no-<{'|'.join(sorted(set(BASE_GROUPS) | set(CHANNEL_GROUPS)))}>"
        )

    if a.crop_sweep:
        argv = [sys.argv[0]]
        if a.scene:
            argv += ["--stem", a.scene]
        if a.crop_variants:
            argv += ["--variants", a.crop_variants]
        if a.crop_baselines:
            argv += ["--baselines", a.crop_baselines]
        if a.crop_pool:
            argv += ["--pool", a.crop_pool]
        if a.crop_seeds:
            argv += ["--seeds", a.crop_seeds]
        if a.crop_skip_baselines:
            argv += ["--skip-baselines"]
        if a.crop_heads:
            argv += ["--heads", a.crop_heads]
        saved, sys.argv = sys.argv, argv
        try:
            from tools.run_crop_sweep import main as crop_main

            raise SystemExit(crop_main())
        finally:
            sys.argv = saved

    if a.verify:
        from tools.verify_assets import main as verify_main

        argv = []
        selected = []
        if a.scene:
            selected = TRAJECTORY_DATASETS[a.scene]
        elif a.paper_order:
            selected = [
                stem
                for dataset in _resolve_paper_order(a.paper_order)
                for stem in TRAJECTORY_DATASETS[dataset]
            ]
        if selected:
            argv += ["--scenes", *(SCENE_NAME[stem] for stem in selected)]
        if a.deep:
            argv.append("--deep")
        if a.json_path:
            argv += ["--json", a.json_path]
        raise SystemExit(verify_main(argv))

    _require_runtime_dependencies(a.stage.split(",") if a.stage else None)

    run_label = a.scene or ("-".join(a.paper_order) if a.paper_order else None)
    init_run_log(run_label)

    order = ["layout", "build", "train", "eval"]
    stages = order if a.stage == "all" else [a.stage]

    if ARCH:
        print(f"[arch] head = {ARCH}")
    if RANDOM_WINDOW:
        print(
            "[data] random-window ablation: 窗口内容打乱,只保留「配置的时序长度」这个形式"
        )
    if THRESHOLD is not None:
        print(
            f"[threshold] eval @ {THRESHOLD:g} m, labels @ {THRESHOLD * LABEL_RATIO:g} m"
        )
    if CHANNELS_ON:
        print(f"[channels] 相对默认的改动: {', '.join(CHANNELS_ON)}")
    _dims = _state_dim_of(CHANNELS_ON)
    print(
        f"[channels] 状态维度 = {_dims}"
        + (" (默认,即论文协议)" if not CHANNELS_ON else "")
    )

    if a.scene is None:
        if a.stage != "all":
            ap.error("--stage requires --scene; paper mode reproduces complete tables")
        paper_order = _resolve_paper_order(a.paper_order)
        if STEP_MAX_MODE == "auto" and len(paper_order) != 1:
            ap.error("--step-max-mode auto requires exactly one --paper-order dataset")
        print("[paper order] " + " -> ".join(paper_order))
        all_failures = []
        DRAFT = (
            ("main table (seed 2026)", phase_table1_draft, "table1"),
            ("10/15 m columns (seed 2026)", phase_table1_thresholds, "table1-th"),
            ("cross-retriever (test)", phase_table2, "table2"),
            ("ablation (first pass)", phase_table3_draft, "table3-draft"),
        )
        _only = {s.strip() for s in a.only.split(",") if s.strip()} if a.only else None
        if _only:
            _known = (
                {k for _t, _f, k in DRAFT}
                | {k for _t, _f, k in _paper_completion_phases(a.table2_retrained)}
                | {"union", "union-mix"}
            )
            _bad = _only - _known
            if _bad:
                ap.error(f"--only 不认识 {sorted(_bad)};可选:{sorted(_known)}")
            print(f"[only] 只跑阶段:{sorted(_only)}")
        for _di, ds in enumerate(paper_order, 1):
            _prog("数据集", _di, len(paper_order), ds)
            print(
                f"\n{'#' * 24} 数据集:{ds}(初稿数据 {_di}/{len(paper_order)}) {'#' * 24}"
            )
            fails = []
            for _pi, (title, fn, _key) in enumerate(DRAFT, 1):
                if _only and _key not in _only:
                    continue
                _prog("阶段", _pi, len(DRAFT), title)
                print(f"\n{'=' * 20} {ds} · {title}({_pi}/{len(DRAFT)}) {'=' * 20}")
                fn(ds, a.force, fails)
            all_failures.extend(fails)
        if not _only or "union" in _only:
            print(f"\n{'#' * 24} 七图通用头(全局) {'#' * 24}")
            phase_union_heads(a.force, all_failures)
        _run_table_replay(datasets=paper_order)
        print("\nDRAFT_TABLES_READY  (三张表初稿数据齐;下面开始补全尾巴)")
        _phases = list(_paper_completion_phases(a.table2_retrained))
        for _pi, (title, fn, _key) in enumerate(_phases, 1):
            if _only and _key not in _only:
                continue
            _prog("阶段", _pi, len(_phases), title)
            for _di, ds in enumerate(paper_order, 1):
                _prog("数据集", _di, len(paper_order), ds)
                print(
                    f"\n{'#' * 24} 数据集:{ds} · {title}(阶段 {_pi}/{len(_phases)},数据集 {_di}/{len(paper_order)}) {'#' * 24}"
                )
                fails = []
                fn(ds, a.force, fails)
                all_failures.extend(fails)
        if not _only or "union-mix" in _only:
            print(f"\n{'#' * 24} 七图混合裁剪头(全局,视野表) {'#' * 24}")
            phase_union_mix(a.force, all_failures)
        _run_table_replay(datasets=paper_order)
        print("\n[done] Paper tables reproduction complete.")
        if all_failures:
            print(
                f"[done] {len(all_failures)} unavailable cells are marked '-' above; "
                "rerun after installing their checkpoints to fill them."
            )
            raise SystemExit(
                f"[reproduce] {len(all_failures)} 个资产没能产出,"
                "退出码非零 —— 别把这次当成跑完了。"
            )
        print("REPRODUCE_ALL_DONE")
        return

    reproduce_trajectory(
        a.scene,
        stages,
        a.force,
        seeds=tuple(x.strip() for x in a.seeds.split(",") if x.strip()),
    )
    print(f"\n[done] {a.scene}: stages {stages} complete.")
    print("REPRODUCE_DONE")


class _TeeStdout:

    def __init__(self, real):
        self._real = real

    def write(self, text):
        self._real.write(text)
        self._real.flush()
        for line in text.splitlines():
            _log_line(line)

    def flush(self):
        self._real.flush()

    def __getattr__(self, name):
        return getattr(self._real, name)


if __name__ == "__main__":
    sys.stdout = _TeeStdout(sys.stdout)
    main()
