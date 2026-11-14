#!/usr/bin/env python


import argparse
import os
import re
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable

VARIANTS = {
    "a3040": {},
    "a4050": {},
    "a5060": {},
    "a7080": {},
    "full": {},
    "mix": {},
    "tilt0": {"CROP_TILT_DEG": "0"},
    "tilt20": {"CROP_TILT_DEG": "20"},
    "centered": {"CROP_CENTERED": "1"},
}


def _symlink_tree(src, dst):
    os.makedirs(dst, exist_ok=True)
    for name in os.listdir(src):
        s, d = os.path.join(src, name), os.path.join(dst, name)
        if os.path.isdir(s):
            _symlink_tree(s, d)
        else:
            if os.path.lexists(d):
                os.remove(d)
            os.symlink(os.path.abspath(os.path.realpath(s)), d)


def _regen_split(src_root, dst_root, split, stem, env_over):

    raw = os.path.join(src_root, f"queries_{split}_raw")
    assert os.path.isdir(raw), f"missing {raw} (build_paper_dataset keeps it)"
    out = os.path.join(dst_root, f"queries_{split}")
    if os.path.isdir(out):
        shutil.rmtree(out)
    env = os.environ.copy()
    env["DATASET"] = stem
    env.update(env_over)
    subprocess.run(
        [PY, "tools/generate_ill_queries.py", "--input", raw, "--output", out],
        cwd=ROOT,
        env=env,
        check=True,
    )
    for dirpath, _, files in os.walk(raw):
        for f in files:
            if f == "pose.txt":
                rel = os.path.relpath(dirpath, raw)
                tgt = os.path.join(out, rel, "pose.txt")
                os.makedirs(os.path.dirname(tgt), exist_ok=True)
                if not os.path.exists(tgt):
                    os.symlink(os.path.abspath(os.path.join(dirpath, f)), tgt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stem")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument(
        "--regen-splits",
        default="test",
        help="comma list of splits regenerated from queries_<split>_raw with "
        "variant crop statistics; the rest stay symlinked to the base scene. "
        "Default 'test' (evaluation-only variant). Use 'train,val,test' for a "
        "variant that will be TRAINED on (e.g. mix) -- symlinked train/val "
        "keep the BASE crop statistics and silently mislabel the zoom channel.",
    )
    a = ap.parse_args()

    from core.config import resolve_dataset

    regen = {s.strip() for s in a.regen_splits.split(",") if s.strip()}
    assert regen <= {"train", "val", "test"}, f"bad --regen-splits: {a.regen_splits}"

    scene = resolve_dataset(a.stem)["name"]
    src_root = os.path.join(ROOT, "datasets", scene)

    for tag in a.variants.split(","):
        env_over = VARIANTS[tag]
        new_scene, new_stem = f"{scene}C{tag}", f"{a.stem}c{tag}"
        dst_root = os.path.join(ROOT, "datasets", new_scene)
        print(f"[{new_scene}] {env_over}  regen={sorted(regen)}")
        _symlink_tree(
            os.path.join(src_root, "database"), os.path.join(dst_root, "database")
        )
        for split in ("train", "val", "test"):
            dst_split = os.path.join(dst_root, f"queries_{split}")
            if split in regen:
                variant_stem = f"{a.stem}c{tag}"
                has_yaml = os.path.exists(
                    os.path.join(ROOT, "configs", "datasets", f"{variant_stem}.yaml")
                )
                _regen_split(
                    src_root,
                    dst_root,
                    split,
                    variant_stem if has_yaml else a.stem,
                    env_over,
                )
            elif os.path.isdir(dst_split):
                print(f"[{new_scene}] queries_{split} 已存在,保持不动")
            elif os.path.isdir(os.path.join(src_root, f"queries_{split}")):
                _symlink_tree(os.path.join(src_root, f"queries_{split}"), dst_split)

        dst_yaml = os.path.join(ROOT, "configs", "datasets", f"{new_stem}.yaml")
        if not os.path.isfile(dst_yaml):
            raise SystemExit(
                f"缺少变体配置 {os.path.relpath(dst_yaml, ROOT)}。\n"
                f"  它是**手工维护**的(git 跟踪),不由本脚本生成。\n"
                f"  照着同目录已有的 robotcarc*.yaml 加一份,再在 jp 上提交同步。"
            )
        print(f"    -> {dst_root}(配置沿用 {os.path.relpath(dst_yaml, ROOT)})")


if __name__ == "__main__":
    main()
