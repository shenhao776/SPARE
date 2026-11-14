#!/usr/bin/env python


import argparse
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core.env_isolation import ensure_reference_scene

ensure_reference_scene()

PY = sys.executable
ANSI = re.compile(r"\x1b\[[0-9;]*m")

THRESHOLD_M = 5.0
DEFAULT_POOL = "MegaLoc_8448"

VARIANT_LABELS = {
    "a4050": "0.4-0.5",
    "a5060": "0.5-0.6",
    "a7080": "0.7-0.8",
    "full": "1.0 (no crop)",
    "a3040": "0.3-0.4 (paper)",
    "tilt0": "no tilt",
    "tilt20": "tilt +-20deg",
    "centered": "centered",
}
DEFAULT_TAGS = ("a4050", "a5060", "a7080", "full")
BASE_LABEL = "0.3-0.4 (paper)"
MIX_TAG = "mix"

_POOL_BACKBONE = {
    "MegaLoc_8448": "MegaLoc",
    "BoQ_dinov2_12288": "BoQ",
    "SALAD_8448": "SALAD",
    "NetVLAD_4096": "NetVLAD",
}

BASELINES = {
    "FoL": ("run_fol_rerank.py", "fol_rerank"),
    "PairVPR-s2": ("run_pairvpr_s2_rerank.py", "pairvpr_s2"),
    "VOP": ("run_vop_rerank.py", "vop_rerank"),
    "R2Former": ("run_r2former_rerank.py", "r2former_rerank"),
}


def sh(cmd, env_extra=None, capture=False):
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    prefix = " ".join(f"{k}={v}" for k, v in (env_extra or {}).items())
    print(f"\n$ {prefix + ' ' if prefix else ''}{' '.join(cmd)}", flush=True)
    if capture:
        out = subprocess.run(
            cmd,
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        print(out.stdout, flush=True)
        return out.returncode, out.stdout
    return subprocess.run(cmd, cwd=ROOT, env=env).returncode, ""


def scene_of(stem_scene, tag):
    return f"{stem_scene}C{tag}"


def ensure_variant(stem, stem_scene, tag, force):

    scene = scene_of(stem_scene, tag)
    have = all(
        os.path.isdir(os.path.join(ROOT, "datasets", scene, f"queries_{s}"))
        for s in ("train", "val", "test")
    )
    if have and not force:
        print(f"[skip] {scene}: 三个 split 都在")
        return True
    code, _ = sh(
        [
            PY,
            "tools/make_crop_variant_scene.py",
            stem,
            "--variants",
            tag,
            "--regen-splits",
            "train,val,test",
        ]
    )
    return code == 0


def ensure_sequence_assets(stem_scene, tag):

    src = os.path.join(ROOT, "assets", "sequence_assets", stem_scene)
    dst = os.path.join(ROOT, "assets", "sequence_assets", scene_of(stem_scene, tag))
    if os.path.islink(dst) and not os.path.exists(dst):
        print(f"[relink] {dst} -> {src}(原链悬空:{os.readlink(dst)})")
        os.remove(dst)
    if os.path.isdir(src) and not os.path.exists(dst):
        os.symlink(src, dst)


def _agg(values):

    if len(values) == 1:
        return f"{values[0]:.1f}"
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return f"{mean:.1f}±{var ** 0.5:.1f}"


def parse_eval(stdout):

    text = ANSI.sub("", stdout)
    block = text.split(f"TEST THRESHOLD = {THRESHOLD_M}m")
    if len(block) < 2:
        return None, None
    raw = ours = None
    for line in block[1].splitlines():
        if "Raw VPR Base" in line and raw is None:
            m = re.search(r"([\d.]+)%", line)
            raw = float(m.group(1)) if m else None
        elif line.lstrip().startswith("| Ours") and ours is None:
            m = re.search(r"([\d.]+)%", line)
            ours = float(m.group(1)) if m else None
    return raw, ours


def baseline_value(slug, scene, pool):
    path = os.path.join(
        ROOT,
        "assets",
        "baseline_results",
        f"{slug}_{scene}__pool-{pool}.json",
    )
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    return data.get("reranked_recall", {}).get(str(int(THRESHOLD_M)))


def exact_raw_recall(scene, pool):

    path = os.path.join(ROOT, "assets", pool, f"test_trajectories_{scene}_no_roi.json")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as handle:
        rows = json.load(handle)
    top = {}
    for r in rows:
        if r["query"] not in top or r["rank"] < top[r["query"]]["rank"]:
            top[r["query"]] = r
    hits = sum(1 for r in top.values() if 0 <= r["error"] <= THRESHOLD_M)
    return 100.0 * hits / len(top) if top else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--stem",
        default="robotcar",
        help="基座场景,stem 或场景名皆可(robotcar / RobotCar / customcampus …)",
    )
    ap.add_argument(
        "--variants",
        default=",".join(DEFAULT_TAGS),
        help="参与出表的裁剪档(逗号分隔),取值见 make_crop_variant_scene.VARIANTS",
    )
    ap.add_argument(
        "--baselines",
        default="FoL,PairVPR-s2",
        help=f"参与对比的学习型重排器,可选 {list(BASELINES)};空串则只跑本方法",
    )
    ap.add_argument(
        "--base-tag",
        default=None,
        help="用该变体场景充当第一列(标 BASE_LABEL),而不是基座场景本身。"
        "视角扫描用 a3040:主表的 RobotCar 不能带视角扰动。",
    )
    ap.add_argument(
        "--reuse-mix",
        action="store_true",
        help="mix 权重已存在时跳过 mix 的重生成/重建/重训,只沿用现成权重"
        "(表的口径就是一份模型、变的是 query 条件)。",
    )
    ap.add_argument(
        "--heads",
        default=None,
        help="'行名=权重场景;行名=权重场景':评测这些**已训好**的头(如七图通用头 "
        "UnionMaps 与七图混合裁剪通用头 UnionMapsMix),不再派生/训练单图的 mix 头。",
    )
    ap.add_argument(
        "--seeds", default="2026,1,2,3,4,5", help="逗号分隔;单个种子则不报 std"
    )
    ap.add_argument(
        "--pool",
        default=DEFAULT_POOL,
        help="候选池(assets/<pool>);换成其他检索器的候选池可检验"
        "「最强检索器上是否也不拖累」,SPARE 权重沿用零样本跨骨干迁移",
    )
    ap.add_argument("--force-regen", action="store_true", help="强制重生成变体图像")
    ap.add_argument(
        "--skip-baselines", action="store_true", help="只出本方法与 Raw 两行"
    )
    a = ap.parse_args()

    from core.config import resolve_dataset
    from tools.make_crop_variant_scene import VARIANTS as KNOWN_VARIANTS

    stem_scene = resolve_dataset(a.stem)["name"]
    global THRESHOLD_M
    THRESHOLD_M = float(
        resolve_dataset(a.stem).get("thresholds", {}).get("test_m", THRESHOLD_M)
    )
    flat_base = THRESHOLD_M != 5.0
    stem = a.stem.lower()
    if not os.path.isfile(os.path.join(ROOT, "configs", "datasets", f"{stem}.yaml")):
        raise SystemExit(
            f"找不到 configs/datasets/{stem}.yaml —— --stem 请给 yaml 的文件名"
            f"(如 robotcar / customcampus),或该场景名对应的 stem"
        )

    tags = [t.strip() for t in a.variants.split(",") if t.strip()]
    bad = [t for t in tags if t not in KNOWN_VARIANTS]
    if bad:
        raise SystemExit(f"未知裁剪档 {bad},可选:{sorted(KNOWN_VARIANTS)}")
    picks = (
        []
        if a.skip_baselines
        else [b.strip() for b in a.baselines.split(",") if b.strip()]
    )
    bad = [b for b in picks if b not in BASELINES]
    if bad:
        raise SystemExit(f"未知基线 {bad},可选:{list(BASELINES)}")

    seeds = [int(x) for x in str(a.seeds).split(",") if str(x).strip()]
    if not seeds:
        raise SystemExit("--seeds 不能为空")
    print(f"[crop-sweep] 种子: {seeds}")
    heads = []
    if a.heads:
        for item in a.heads.split(";"):
            label, _, spec = item.partition("=")
            wscene, _, over = spec.partition("@")
            if not label.strip() or not wscene.strip():
                raise SystemExit(
                    f"--heads 格式是 '行名=权重场景[@KEY=VAL,...];...',收到 {item!r}"
                )
            overrides = {}
            for kv in filter(None, (x.strip() for x in over.split(","))):
                k, _, v = kv.partition("=")
                if not k.startswith("ABLATION_") or v not in ("True", "False"):
                    raise SystemExit(
                        f"--heads 的通道覆盖只接受 ABLATION_*=True/False,收到 {kv!r}"
                    )
                overrides[k] = v
            heads.append((label.strip(), wscene.strip(), overrides))
        print(f"[crop-sweep] 评测已训好的头: {heads}")

    failures = []
    base_col = (a.base_tag, BASE_LABEL) if a.base_tag else (None, BASE_LABEL)
    columns = [base_col] + [(t, VARIANT_LABELS[t]) for t in tags]
    if a.base_tag and a.base_tag not in tags:
        tags = [a.base_tag] + tags

    for tag in tags + ([] if (a.reuse_mix or heads) else [MIX_TAG]):
        if not ensure_variant(stem, stem_scene, tag, a.force_regen):
            failures.append(f"variant:{tag}")
            continue
        ensure_sequence_assets(stem_scene, tag)
        build_env = {"BUILD_WORKERS": os.environ.get("BUILD_WORKERS", "3")}
        build_cmd = [
            PY,
            "reproduce.py",
            "--scene",
            scene_of(stem_scene, tag),
            "--stage",
            "build",
        ]
        if a.pool != DEFAULT_POOL:
            build_cmd += ["--model", _POOL_BACKBONE.get(a.pool, a.pool)]
        if tag != MIX_TAG:
            build_cmd += ["--splits", "test"]
        code, _ = sh(build_cmd, build_env)
        if code:
            failures.append(f"build:{tag}")

    mix_scene = scene_of(stem_scene, MIX_TAG)
    reproduce_entry = (
        "pipeline/reproduce.py"
        if os.path.exists("pipeline/reproduce.py")
        else "reproduce.py"
    )
    train_cmd = [
        PY,
        reproduce_entry,
        "--scene",
        mix_scene,
        "--stage",
        "train",
        "--seeds",
        ",".join(str(s) for s in seeds),
    ]
    if a.pool != DEFAULT_POOL:
        train_cmd += ["--model", _POOL_BACKBONE.get(a.pool, a.pool)]
    if heads:
        print("[heads] 评测给定的头,跳过单图 mix 的训练", flush=True)
    elif a.reuse_mix:
        print("[reuse-mix] 沿用现成 mix 权重,跳过训练", flush=True)
    else:
        code, _ = sh(train_cmd)
        if code:
            failures.append("train:mix")

    raw_label = f"Raw {_POOL_BACKBONE.get(a.pool, a.pool)}"
    head_list = heads or [("SPARE (mix)", mix_scene, {})]
    rows = {raw_label: {}}
    rows.update({label: {} for label, _, _ in head_list})
    for (head_label, weight_scene, head_over), (tag, label) in [
        (h, c) for h in head_list for c in columns
    ]:
        scene = stem_scene if tag is None else scene_of(stem_scene, tag)
        raws, ours_all = [], []
        for seed in seeds:
            env = {
                "DATASET": scene.lower(),
                "MODEL": "megaloc",
                "VPR_MODEL_TYPE": "MegaLoc",
                "TRAIN_SEED": str(seed),
                "ENABLE_SCENE_MODELS_CACHE": "True",
                "STAR_WEIGHT_SCENE": weight_scene,
                **head_over,
            }
            if a.pool != DEFAULT_POOL:
                env["VPR_MODEL_TYPE"] = _POOL_BACKBONE.get(a.pool, a.pool)
                env["MODEL"] = env["VPR_MODEL_TYPE"].lower()
            code, out = sh([PY, "tools/fast_offline_eval.py"], env, capture=True)
            raw, ours = parse_eval(out)
            if code or raw is None or ours is None:
                failures.append(f"eval:{head_label}:{tag or 'base'}:seed{seed}")
                continue
            raws.append(raw)
            ours_all.append(ours)
        if not ours_all:
            continue
        exact = exact_raw_recall(scene, a.pool)
        rows[raw_label][label] = f"{exact if exact is not None else raws[0]:.1f}"
        rows[head_label][label] = _agg(ours_all)

    for name in picks:
        script, slug = BASELINES[name]
        rows[name] = {}
        for tag, label in columns:
            scene = stem_scene if tag is None else scene_of(stem_scene, tag)
            cmd = [PY, f"tools/{script}", "--scenes", scene, "--pool", a.pool]
            env_b = None
            if flat_base:
                cmd += ["--success-thresholds", "10,15,25"]
                env_b = {"DATASET": scene.lower()}
            code, _ = sh(cmd, env_b)
            value = baseline_value(slug, scene, a.pool)
            if code or value is None:
                failures.append(f"{name}:{tag or 'base'}")
                continue
            rows[name][label] = f"{value:.1f}"

    order = [raw_label] + picks + [label for label, _, _ in head_list]
    headers = [label for _, label in columns]
    stem_out = "crop_sweep" if stem == "robotcar" else f"crop_sweep_{stem}"
    if a.pool != DEFAULT_POOL:
        stem_out += f"_{a.pool}"
    out_path = os.path.join(ROOT, "results", "tables", f"{stem_out}.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write("method," + ",".join(headers) + "\n")
        for name in order:
            cells = [
                rows[name][h] if h in rows.get(name, {}) else "--" for h in headers
            ]
            handle.write(f"{name}," + ",".join(cells) + "\n")

    print(f"\n=== 视野鲁棒性表(R@1 @ {THRESHOLD_M} m, pool={a.pool}) ===")
    print(f"{'method':<22}" + "".join(f"{h:>16}" for h in headers))
    for name in order:
        cells = "".join(
            f"{rows[name][h]:>16}" if h in rows.get(name, {}) else f"{'--':>16}"
            for h in headers
        )
        print(f"{name:<22}{cells}")
    print(f"\n-> {os.path.relpath(out_path, ROOT)}")

    if failures:
        print("\n未完成的格:" + ", ".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
