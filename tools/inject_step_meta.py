#!/usr/bin/env python


import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _shape(seq, L):

    import math

    v = [float(x) for x in (seq or [])][:L]
    if not v or max(v) <= 0:
        return [0.0] * L, [0.0] * L
    c = L // 2
    off = (v.index(max(v)) - c) / float(c or 1)
    srt = sorted(v)
    med = srt[len(srt) // 2]
    sharp = math.log1p(max(v) / med) if med > 1e-9 else math.log1p(max(v))
    return [off] * L, [sharp] * L


FIELDS = (
    "step_offset_m",
    "step_turn_deg",
    "step_change",
    "step_valid",
    "peak_off",
    "peak_sharp",
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--model-id", default="MegaLoc_8448")
    ap.add_argument("--splits", default="train,val,test")
    a = ap.parse_args()

    os.environ.setdefault("DATASET", a.scene)
    from core.config import (
        TARGET_SCENE,
        SPATIAL_STEPS,
        data_tag,
        DATA_ABLATION_RANDOM_WINDOW,
        STEP_FEATURE_TAU,
        STEP_MAX_DIST,
        STEP_BUDGET,
    )
    from core.vpr_core import (
        pose_manager,
        get_all_images_recursive,
        get_database_spatial_sequence,
    )

    db_dir = os.path.join(ROOT, "datasets", TARGET_SCENE, "database")
    by_name = {os.path.basename(p): p for p in get_all_images_recursive(db_dir)}
    L = sum(SPATIAL_STEPS) + 1
    step_meta_provenance = {
        "fields": list(FIELDS),
    }
    print(
        f"[{TARGET_SCENE}] L={L} tau={STEP_FEATURE_TAU} max_dist={STEP_MAX_DIST} "
        f"budget={STEP_BUDGET}"
    )

    pose_cache, win_cache = {}, {}

    def pose(p):
        if p not in pose_cache:
            info = pose_manager.get_pose_info(p)
            pose_cache[p] = info["global"] if info and info.get("global") else None
        return pose_cache[p]

    def window(path):
        if path not in win_cache:
            try:
                win_cache[path] = get_database_spatial_sequence(
                    path, pose_manager, SPATIAL_STEPS, return_meta=True
                )
            except Exception:
                win_cache[path] = None
        return win_cache[path]

    for split in [s.strip() for s in a.splits.split(",") if s.strip()]:
        path = os.path.join(
            ROOT,
            "assets",
            a.model_id,
            f"{split}_trajectories_{TARGET_SCENE}{data_tag()}.json",
        )
        if not os.path.isfile(path):
            print(f"[skip] {split}: {os.path.relpath(path, ROOT)} not found")
            continue
        from sl_agent.cache_fingerprint import injector_current

        if injector_current(path, "step_meta", step_meta_provenance, expected_steps=L):
            print(f"[skip] {split}: step_meta already current")
            continue
        with open(path, "r", encoding="utf-8") as fh:
            rows = json.load(fh)

        rw_db_list = rw_by_name = rw_qpaths = None
        if DATA_ABLATION_RANDOM_WINDOW:
            from sl_agent.random_window_replay import (
                load_db_rel_paths,
                replay,
                verify_row,
            )

            rw_db_list, rw_by_name = load_db_rel_paths(a.model_id, TARGET_SCENE)
            qdir = os.path.join(ROOT, "datasets", TARGET_SCENE, f"queries_{split}")
            rw_qpaths = {
                os.path.splitext(os.path.basename(pp))[0]: pp
                for pp in get_all_images_recursive(qdir)
            }

        filled = zeroed = rw_verified = 0
        pads = []
        for row in rows:
            if DATA_ABLATION_RANDOM_WINDOW:
                rel_p = rw_by_name.get(row.get("candidate_name"))
                q_path = rw_qpaths.get(row.get("query"))
                q_g = pose(q_path) if q_path else None
                if rel_p is None or q_path is None:
                    for f in FIELDS:
                        row[f] = [0.0] * L
                    row["peak_off"], row["peak_sharp"] = _shape(row.get("inliers"), L)
                    zeroed += 1
                    continue
                seq_len = len(row.get("step_errors") or []) or L
                rw_paths = replay(
                    TARGET_SCENE, rw_db_list, row["query"], rel_p, seq_len
                )
                n_bad = verify_row(
                    row, rw_paths, q_path, q_g, pose_manager, TARGET_SCENE
                )
                if n_bad:
                    raise SystemExit(
                        f"[{TARGET_SCENE}/{split}] 随机窗口重放与缓存不符:"
                        f"query={row['query']} cand={row['candidate_name']} "
                        f"{n_bad}/{seq_len} 步的 step_errors 对不上 —— "
                        f"拒绝写入(配错帧的通道不会报错,只会给出错的数)"
                    )
                rw_verified += 1
                frames = [
                    os.path.join(ROOT, "datasets", TARGET_SCENE, rp) for rp in rw_paths
                ]
                centre = pose(os.path.join(ROOT, "datasets", TARGET_SCENE, rel_p))
                meta = [{"change": 0.0, "valid": True}] * len(frames)
            else:
                cand = by_name.get(row.get("candidate_name"))
                out = window(cand) if cand else None
                centre = pose(cand) if cand else None
                if not out or len(out) != 3 or centre is None:
                    for f in FIELDS:
                        row[f] = [0.0] * L
                    row["peak_off"], row["peak_sharp"] = _shape(row.get("inliers"), L)
                    zeroed += 1
                    continue
                back, fwd, meta = out
                frames = list(reversed(back)) + [cand] + list(fwd)
            if centre is None:
                for f in FIELDS:
                    row[f] = [0.0] * L
                row["peak_off"], row["peak_sharp"] = _shape(row.get("inliers"), L)
                zeroed += 1
                continue
            mid = len(frames) // 2
            offs, turns, chg, valid = [], [], [], []
            for i, frame in enumerate(frames):
                g = pose(frame)
                sign = -1.0 if i < mid else 1.0
                if g is None:
                    offs.append(0.0)
                    turns.append(0.0)
                else:
                    d = pose_manager.calculate_physical_distance(frame, centre, g)
                    t = pose_manager.calculate_angular_distance(centre, g)
                    offs.append(sign * float(abs(d or 0.0)))
                    turns.append(sign * float(t or 0.0))
                m = meta[i] if i < len(meta) else {}
                chg.append(float(m.get("change") or 0.0))
                valid.append(1.0 if m.get("valid", True) else 0.0)
            row["step_offset_m"], row["step_turn_deg"] = offs, turns
            row["step_change"], row["step_valid"] = chg, valid
            row["peak_off"], row["peak_sharp"] = _shape(row.get("inliers"), L)
            pads.append(1.0 - sum(valid) / max(len(valid), 1))
            filled += 1
        if DATA_ABLATION_RANDOM_WINDOW and rw_verified:
            print(
                f"[{split}] 随机窗口重放:{rw_verified} 行全部通过 step_errors 逐位校验"
            )

        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(rows, fh)
        os.replace(tmp, path)
        pad = 100.0 * (sum(pads) / len(pads)) if pads else 0.0
        print(
            f"[{split}] {len(rows)} rows: {filled} replayed, {zeroed} zero-filled; "
            f"padding {pad:.1f}% of steps -> {os.path.relpath(path, ROOT)}"
        )
        if pad < 0.1:
            print(
                "        (no padding anywhere: step_valid is constant, so "
                "normalize_block will zero it -- harmless but it carries nothing)"
            )

        from sl_agent.cache_fingerprint import refresh_output_fingerprint

        refresh_output_fingerprint(path, step_meta=step_meta_provenance)
    return 0


if __name__ == "__main__":
    sys.exit(main())
