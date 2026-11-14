#!/usr/bin/env python


import argparse
import hashlib
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("DATASET", "robotcar")

from sl_agent.cache_fingerprint import read_fingerprint, write_fingerprint

PY = sys.executable


def _member_paths(model_id, split, scenes):
    return [
        os.path.join(
            ROOT, "assets", model_id, f"{split}_trajectories_{scene}_no_roi.json"
        )
        for scene in scenes
    ]


def _merge(model_id, split, scenes, tag):

    out = os.path.join(
        ROOT, "assets", model_id, f"{split}_trajectories_{tag}_no_roi.json"
    )
    members = _member_paths(model_id, split, scenes)
    stamps = []
    for path in members:
        stamp = read_fingerprint(path) or {}
        key = stamp.get("cache_key") or stamp.get("semantic_digest")
        if not stamp.get("complete") or not key:
            raise SystemExit(f"member cache lacks a complete fingerprint: {path}")
        settings = stamp.get("settings") or {}
        if int(settings.get("expected_steps", 0)) <= 1:
            raise SystemExit(
                f"{path}: L = 1 flat cache cannot join a window-protocol union"
            )
        stamps.append((path, stamp))
    merged_key = hashlib.sha256(
        json.dumps(
            {
                "tag": tag,
                "members": [
                    (
                        os.path.basename(p),
                        (
                            s["cache_key"]
                            if s.get("cache_key")
                            else s.get("semantic_digest")
                        ),
                    )
                    for p, s in stamps
                ],
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    optional = ("vprpatch",)
    carried = [
        name
        for name in optional
        if all(name in (s.get("injectors") or {}) for _p, s in stamps)
    ]
    existing = read_fingerprint(out) or {}
    if (
        existing.get("complete")
        and existing.get("cache_key") == merged_key
        and os.path.isfile(out)
        and all(
            (existing.get("injectors") or {}).get(name)
            == stamps[0][1]["injectors"][name]
            for name in carried
        )
    ):
        print(
            f"[skip] {os.path.basename(out)}: already merged from the same member caches"
        )
        return out
    required = ("dino", "step_meta", "zoom")
    member_injectors = {}
    for path, stamp in stamps:
        present = stamp.get("injectors") or {}
        missing = [name for name in required if name not in present]
        if missing:
            raise SystemExit(
                f"{path}: injected channels missing {missing}; run the injectors first"
            )
        member_injectors[os.path.basename(path)] = present
    injectors = {name: stamps[0][1]["injectors"][name] for name in required}
    for name in carried:
        values = {json.dumps(s["injectors"][name], sort_keys=True) for _p, s in stamps}
        if len(values) != 1:
            raise SystemExit(
                f"members disagree on the {name} injector provenance: {sorted(values)}"
            )
        injectors[name] = stamps[0][1]["injectors"][name]
    rows, n_queries = [], 0
    for scene, (path, stamp) in zip(scenes, stamps):
        with open(path, encoding="utf-8") as fh:
            part = json.load(fh)
        for row in part:
            row["query"] = f"{scene}::{row['query']}"
        rows.extend(part)
        n_queries += int(stamp.get("n_queries") or len({r["query"] for r in part}))
        print(f"  + {scene:16s} {len(part):7d} rows")
    tmp = f"{out}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False)
    os.replace(tmp, out)
    settings = dict(stamps[0][1].get("settings") or {})
    settings["merged_from"] = [os.path.basename(p) for p, _ in stamps]
    settings["member_injectors"] = member_injectors
    write_fingerprint(
        out,
        {
            "complete": True,
            "cache_key": merged_key,
            "semantic_digest": merged_key,
            "n_queries": n_queries,
            "injectors": injectors,
            "settings": settings,
            "structure_schema": stamps[0][1].get("structure_schema", 1),
        },
    )
    print(f"[merged] {os.path.basename(out)}: {len(rows)} rows, {n_queries} queries")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--scenes", required=True, help="comma-separated member scene names"
    )
    ap.add_argument(
        "--tag",
        required=True,
        help="virtual scene name, e.g. UnionMaps or LOO_RobotCar",
    )
    ap.add_argument("--seeds", default="2026")
    ap.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="success radius (m); default = protocol 5 m",
    )
    ap.add_argument("--model-id", default="MegaLoc_8448")
    ap.add_argument(
        "--dataset", default="robotcar", help="yaml stem carrying the shared protocol"
    )
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="protocol override for an ablation arm, e.g. --set ABLATION_USE_DINO=False "
        "--set ABLATION_ARCH=gru (applied after paper_protocol_env; weight name follows agent_tag)",
    )
    a = ap.parse_args()
    scenes = [s.strip() for s in a.scenes.split(",") if s.strip()]
    if not all(ch.isalnum() or ch in "-_" for ch in a.tag):
        raise SystemExit("--tag may contain only letters, digits, '-' and '_'")

    for split in ("train", "val"):
        _merge(a.model_id, split, scenes, a.tag)

    from core.config import paper_protocol_env

    base_env = dict(os.environ)
    base_env.update(paper_protocol_env())
    for item in a.set:
        key, _, value = item.partition("=")
        if not key.startswith(("ABLATION_", "DATA_ABLATION_")) or not value:
            raise SystemExit(
                f"--set expects ABLATION_*/DATA_ABLATION_* KEY=VALUE, got {item!r}"
            )
        base_env[key] = value
    base_env.update(
        {
            "DATASET": a.dataset,
            "MODEL": "megaloc",
            "VPR_MODEL_TYPE": "MegaLoc",
            "ENABLE_SCENE_MODELS_CACHE": "True",
            "TRAIN_SCENE_TAG": a.tag,
            "FORCE_RETRAIN": "True" if a.force else "False",
        }
    )
    if a.threshold is not None:
        base_env["SUCCESS_THRESHOLD_TEST"] = f"{a.threshold:g}"
        base_env["SUCCESS_THRESHOLD_TRAIN"] = f"{a.threshold * 1.6:g}"
    for seed in [s.strip() for s in a.seeds.split(",") if s.strip()]:
        env = dict(base_env)
        env["TRAIN_SEED"] = seed
        print(
            f"\n# train_agent: tag={a.tag} seed={seed} threshold={a.threshold or 'protocol'}"
        )
        subprocess.run(
            [PY, os.path.join(ROOT, "train_agent.py")], env=env, check=True, cwd=ROOT
        )


if __name__ == "__main__":
    main()
