#!/usr/bin/env python

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from core.paper_config import load_paper_config
from tools.run_crop_sweep import ensure_sequence_assets

PY = sys.executable
_PAPER = load_paper_config()
MEMBERS = [
    (_PAPER["all_scene_stems"][scene], scene)
    for scene in _PAPER["default_head"]["members"]
]
SEEDS = ",".join(str(seed) for seed in _PAPER["seeds"])
MIX_TAG = _PAPER["union_heads"]["mix_tag"]
MIX_SUFFIX = _PAPER["union_heads"]["mix_suffix"]


def run(cmd, env_extra=None):
    print("$", " ".join(cmd), flush=True)
    env = dict(os.environ, **(env_extra or {}))
    code = subprocess.run(cmd, cwd=ROOT, env=env).returncode
    if code:
        raise SystemExit(f"failed ({code}): {' '.join(cmd)}")


def main():
    for stem, scene in MEMBERS:
        variant = f"{scene}{MIX_SUFFIX}"
        have = all(
            os.path.isdir(os.path.join(ROOT, "datasets", variant, f"queries_{s}"))
            for s in ("train", "val")
        )
        if have:
            print(f"[skip] {variant}: queries_train/val 已在")
        else:
            run(
                [
                    PY,
                    "tools/make_crop_variant_scene.py",
                    stem,
                    "--variants",
                    "mix",
                    "--regen-splits",
                    "train,val",
                ],
                {"DATASET": stem},
            )
        ensure_sequence_assets(scene, "mix")
        workers = "1" if scene == "Boreas" else "2"
        run(
            [
                PY,
                "reproduce.py",
                "--scene",
                variant,
                "--stage",
                "build",
                "--splits",
                "train,val",
            ],
            {"BUILD_WORKERS": workers},
        )
    run(
        [
            PY,
            "tools/train_union_head.py",
            "--scenes",
            ",".join(f"{s}{MIX_SUFFIX}" for _, s in MEMBERS),
            "--tag",
            MIX_TAG,
            "--seeds",
            SEEDS,
        ]
    )
    print("[union-mix] done")


if __name__ == "__main__":
    main()
