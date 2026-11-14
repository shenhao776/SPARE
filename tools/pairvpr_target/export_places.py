import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--n-val", type=int, default=150)
    ap.add_argument("--model-id", default="MegaLoc_8448")
    a = ap.parse_args()

    from core.config import TOP_K, data_tag
    from core.vpr_core import get_all_images_recursive, pose_manager

    def rel(p):
        return os.path.relpath(p, ROOT)

    def located(paths):
        out = []
        for p in paths:
            info = pose_manager.get_pose_info(p)
            if info is None:
                raise SystemExit(f"[export] no pose for {p}")
            out.append([rel(p), float(info["local"][0]), float(info["local"][1])])
        return out

    qdir = os.path.join(ROOT, "datasets", a.scene, "queries_train")
    ddir = os.path.join(ROOT, "datasets", a.scene, "database")
    queries = located(get_all_images_recursive(qdir))
    database = located(get_all_images_recursive(ddir))

    cache = f"assets/{a.model_id}/val_trajectories_{a.scene}{data_tag()}.json"
    rows = json.load(open(cache, encoding="utf-8"))
    by_q = {}
    for r in rows:
        by_q.setdefault(str(r["query"]), []).append(r)
    vstem = {
        os.path.splitext(os.path.basename(p))[0]: p
        for p in get_all_images_recursive(
            os.path.join(ROOT, "datasets", a.scene, "queries_val")
        )
    }
    dstem = {os.path.splitext(os.path.basename(p[0]))[0]: p[0] for p in database}
    val = []
    for q, cands in list(by_q.items())[: a.n_val]:
        cands = sorted(cands, key=lambda x: x.get("rank", 999))[:TOP_K]
        val.append(
            {
                "query": rel(vstem[q]),
                "cands": [
                    dstem[os.path.splitext(c["candidate_name"])[0]] for c in cands
                ],
                "err": [float(c["error"]) for c in cands],
            }
        )
    os.makedirs("assets/pairvpr_target", exist_ok=True)
    out = f"assets/pairvpr_target/places_{a.scene}.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"queries": queries, "database": database, "val": val}, fh)
    print(
        f"[export] {a.scene}: {len(queries)} queries, {len(database)} db, {len(val)} val -> {out}"
    )


if __name__ == "__main__":
    main()
