#!/usr/bin/env python


import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable

BUILDER = {
    "train": "sl_agent/build_sl_dataset_train.py",
    "val": "sl_agent/build_sl_dataset_val.py",
    "test": "sl_agent/build_sl_dataset_test.py",
}
QDIR = {"train": "queries_train", "val": "queries_val", "test": "queries_test"}


def official_json(scene, model_id, split):

    from core.config import data_tag

    return os.path.join(
        ROOT, "assets", model_id, f"{split}_trajectories_{scene}{data_tag()}.json"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", required=True, choices=list(BUILDER))
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument(
        "--model",
        default=None,
        help="backbone; defaults to VPR_MODEL_TYPE / MODEL from the "
        "environment. This used to default to the literal "
        "'MegaLoc' and then OVERWRITE the environment with it, so "
        "a caller that exported VPR_MODEL_TYPE=SALAD but passed no "
        "--model silently rebuilt MegaLoc instead. An entire "
        "cross-backbone pass did exactly that for 2.5 hours.",
    )
    ap.add_argument(
        "--out-suffix",
        default="",
        help="append to the merged filename (e.g. '.par' for a "
        "byte-compare run against the serial official json)",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="rebuild even when the fingerprint says the merged "
        "cache still describes the current inputs",
    )
    a = ap.parse_args()

    if a.workers < 1:
        ap.error("--workers must be at least 1")

    a.model = (
        a.model
        or os.environ.get("VPR_MODEL_TYPE")
        or os.environ.get("MODEL")
        or "MegaLoc"
    )
    env_base = dict(
        os.environ,
        DATASET=a.dataset,
        MODEL=a.model,
        VPR_MODEL_TYPE=a.model,
    )
    from core.config import resolve_dataset

    cfg = resolve_dataset(a.dataset)
    scene = cfg["name"]
    probe = subprocess.run(
        [PY, "-c", "from core.config import VPR_MODEL_ID; print(VPR_MODEL_ID)"],
        env=env_base,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    if probe.returncode != 0 or not probe.stdout.strip():
        sys.stderr.write(probe.stdout + probe.stderr)
        raise SystemExit("could not resolve the requested backbone/model id")
    model_id = probe.stdout.strip().splitlines()[-1]
    target = official_json(scene, model_id, a.split) + a.out_suffix

    def resume_check(stage):

        p = subprocess.run(
            [PY, BUILDER[a.split]],
            env=dict(env_base, BUILD_RESUME_CHECK="1"),
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        if p.returncode not in (0, 10):
            sys.stderr.write(p.stdout + p.stderr)
            raise SystemExit(f"resume check ({stage}) failed rc={p.returncode}")
        return p.returncode == 0, p.stdout.strip().splitlines()

    if not a.out_suffix and not a.force:
        ok, lines = resume_check("before")
        if ok:
            print(
                f"[parallel] {scene}/{a.split}: "
                + (lines[-1] if lines else "cache matches the current inputs")
            )
            return
        print(
            f"[parallel] {scene}/{a.split}: "
            + (lines[-1] if lines else "rebuild required")
        )

    subprocess.run(
        [
            PY,
            "-c",
            (
                "from core.vpr_core import VPRService; s=VPRService();"
                f"s._get_or_compute_db_features('datasets/{scene}')"
            ),
        ],
        env=env_base,
        cwd=ROOT,
        check=True,
    )

    t0 = time.time()
    procs, shard_files = [], []
    for i in range(a.workers):
        env = dict(env_base, BUILD_SHARD=f"{i}/{a.workers}")
        shard_files.append(
            official_json(scene, model_id, a.split) + f".shard{i}of{a.workers}"
        )
        if os.path.exists(shard_files[-1]):
            os.remove(shard_files[-1])
        procs.append(subprocess.Popen([PY, BUILDER[a.split]], env=env, cwd=ROOT))
    rc = [p.wait() for p in procs]
    if any(rc):
        raise SystemExit(f"worker exit codes {rc}; shards kept for inspection")

    import core.vpr_core as vc

    os.environ.update(DATASET=a.dataset, VPR_MODEL_TYPE=a.model)
    qdir = os.path.join(ROOT, "datasets", scene, QDIR[a.split])
    order = vc.get_all_images_recursive(qdir)
    stem = lambda p: os.path.basename(p).split(".")[0]
    order_keys = [stem(p) for p in order]

    seen, by_query = set(), {}
    for sf in shard_files:
        for row in json.load(open(sf, encoding="utf-8")):
            key = (row["query"], row["candidate_name"])
            if key in seen:
                continue
            seen.add(key)
            by_query.setdefault(row["query"], []).append(row)
    merged = []
    for k in order_keys:
        merged.extend(by_query.get(k, ()))
    unknown = set(by_query) - set(order_keys)
    assert not unknown, f"rows for unknown queries: {sorted(unknown)[:3]}"

    from core.config import SPATIAL_STEPS, TOP_K
    from sl_agent.cache_fingerprint import clone_rows_stat, structural_error

    counts = sorted({len(v) for v in by_query.values()})
    probe = {
        "rows": len(merged),
        "queries": set(by_query),
        "rows_per_query": counts,
        "step_lengths": sorted(
            {
                len(row.get(f) or [])
                for row in merged
                for f in (
                    "inliers",
                    "scores",
                    "step_errors",
                    "area_step",
                    "area_query",
                    "scale_ratio",
                )
            }
        ),
        "degenerate": sum(
            1
            for row in merged
            if len(row.get("step_errors") or []) > 1
            and len(set(row["step_errors"])) == 1
        ),
        "clone_groups": sum(clone_rows_stat(v) for v in by_query.values()),
        "groups": len(by_query),
    }
    broken = structural_error(
        probe, {"top_k": TOP_K, "expected_steps": sum(SPATIAL_STEPS) + 1}
    )
    if broken:
        raise SystemExit(
            f"[parallel] refusing to write {os.path.basename(target)}: {broken}\n"
            f"           shards kept for inspection: {shard_files}"
        )

    target_tmp = f"{target}.tmp.{os.getpid()}"
    with open(target_tmp, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    os.replace(target_tmp, target)
    for sf in shard_files:
        for path in (sf, sf + ".meta.json"):
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
    print(
        f"[parallel] {scene}/{a.split}: {len(merged)} rows from "
        f"{a.workers} workers in {time.time()-t0:.0f}s -> {target}"
    )

    if not a.out_suffix:
        from sl_agent.cache_fingerprint import (
            build_fingerprint,
            state_build_settings,
            write_fingerprint,
        )

        write_fingerprint(
            target,
            build_fingerprint(order, state_build_settings(a.split)),
        )
        ok, lines = resume_check("after")
        if not ok:
            raise SystemExit(
                "merged cache does not match the requested inputs: "
                + (lines[-1] if lines else "unknown")
            )


if __name__ == "__main__":
    main()
