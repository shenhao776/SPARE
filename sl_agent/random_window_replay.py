import os
import random

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_db_rel_paths(model_id, scene):

    p = os.path.join(ROOT, "assets", model_id, f"database_features_{scene}.npz")
    d = np.load(p, allow_pickle=True)
    rel = [str(x) for x in d["paths"]]
    by_name = {os.path.basename(x): x for x in rel}
    if len(by_name) != len(rel):
        raise SystemExit(f"[{scene}] npz 里有重名库帧,basename 无法唯一定位候选")
    return rel, by_name


def replay(scene, db_rel_paths_list, q_name, rel_p, seq_len):

    rng = random.Random(f"window-v2|{scene}|{q_name}|{rel_p}")
    paths = rng.sample(db_rel_paths_list, seq_len)
    paths[seq_len // 2] = rel_p
    return paths


def verify_row(row, paths, q_path, q_global, pose_manager, scene):

    want = row.get("step_errors") or []
    if len(want) != len(paths):
        return max(len(want), len(paths))
    bad = 0
    for i, rp in enumerate(paths):
        sp_abs = os.path.join(ROOT, "datasets", scene, rp)
        info = pose_manager.get_pose_info(sp_abs)
        g = info["global"] if info and info.get("global") else None
        if g is None or q_global is None:
            got = -1.0
        else:
            d = pose_manager.calculate_physical_distance(q_path, q_global, g)
            got = round(d, 3) if d is not None else -1.0
        if abs(float(want[i]) - got) > 1e-6:
            bad += 1
    return bad
