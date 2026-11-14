#!/usr/bin/env python


import argparse
import json
import os
import random
import sys
import zlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FIELD = "zoom"


def _expected_steps():

    from core.config import SPATIAL_STEPS

    return sum(SPATIAL_STEPS) + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--model-id", default="MegaLoc_8448")
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument(
        "--optical",
        type=float,
        default=3.2,
        help="zoom for full-frame recordings -- a REAL optical "
        "zoom (CustomCampus phone: 3.2x). The channel means "
        "'how much narrower is the query FOV than a database "
        "frame', so for two different physical cameras it "
        "would instead be tan(map_hfov/2)/tan(query_hfov/2).",
    )
    ap.add_argument(
        "--shuffle",
        type=int,
        default=None,
        metavar="SEED",
        help="negative control: permute the per-query zoom values "
        "with this seed and write them to `zoom_shuf` instead. "
        "Same distribution, same per-query-constant structure, "
        "wrong pairing -- so an arm that still gains is gaining "
        "from having one more channel, not from the zoom. "
        "⚠️ meaningless where a split's zoom is already constant "
        "(a real optical-zoom recording): nothing to permute.",
    )
    a = ap.parse_args()

    os.environ.setdefault("DATASET", a.scene)
    from core.config import TARGET_SCENE, SPATIAL_STEPS, CFG, data_tag
    from core.vpr_core import get_all_images_recursive

    import importlib.util

    def _gen_for(split):
        _spec = importlib.util.spec_from_file_location(
            "_gen", os.path.join(ROOT, "tools", "generate_ill_queries.py")
        )
        _gen = importlib.util.module_from_spec(_spec)
        _saved = os.environ.get("QUERY_SPLIT")
        os.environ["QUERY_SPLIT"] = split
        try:
            _spec.loader.exec_module(_gen)
        finally:
            if _saved is None:
                os.environ.pop("QUERY_SPLIT", None)
            else:
                os.environ["QUERY_SPLIT"] = _saved
        return _gen

    _gen = _gen_for("train")
    lo, hi = _gen.MIN_CROP_RATIO, _gen.MAX_CROP_RATIO
    crop_seed = _gen.CROP_SEED
    full_frame = set(getattr(CFG.dataset.build, "full_frame_recordings", []) or [])
    L = sum(SPATIAL_STEPS) + 1
    provenance_name = "zoom_shuf" if a.shuffle is not None else "zoom"
    zoom_provenance = {
        "crop_seed": crop_seed,
        "range": [lo, hi],
        "optical": a.optical,
        "shuffle": a.shuffle,
    }
    print(
        f"[{TARGET_SCENE}] crop ratio U({lo}, {hi})  CROP_SEED={crop_seed}  "
        f"full-frame recordings: {sorted(full_frame) or 'none'}"
    )

    root = os.path.join(ROOT, "datasets", TARGET_SCENE)
    stem_to_rel = {}
    for split in ("train", "val", "test"):
        d = os.path.join(root, f"queries_{split}")
        if not os.path.isdir(d):
            continue
        for p in get_all_images_recursive(d):
            rel = os.path.relpath(p, d)
            stem_to_rel[os.path.splitext(os.path.basename(p))[0]] = (rel, p)

    for split in [s.strip() for s in a.splits.split(",") if s.strip()]:
        _g = _gen_for(split)
        lo, hi, crop_seed = _g.MIN_CROP_RATIO, _g.MAX_CROP_RATIO, _g.CROP_SEED
        zoom_provenance = {**zoom_provenance, "crop_seed": crop_seed, "range": [lo, hi]}
        path = os.path.join(
            ROOT,
            "assets",
            a.model_id,
            f"{split}_trajectories_{TARGET_SCENE}{data_tag()}.json",
        )
        if not os.path.isfile(path):
            print(f"[skip] {split}: not found")
            continue
        from sl_agent.cache_fingerprint import injector_current

        if injector_current(
            path, provenance_name, zoom_provenance, expected_steps=_expected_steps()
        ):
            print(f"[skip] {split}: {provenance_name} already current")
            continue
        with open(path, "r", encoding="utf-8") as fh:
            rows = json.load(fh)
        vals, miss = [], 0
        for row in rows:
            rel_p = stem_to_rel.get(row.get("query"))
            if rel_p is None:
                row[FIELD] = [0.0] * L
                miss += 1
                continue
            rel, full = rel_p
            if any(fr in full for fr in full_frame):
                z = a.optical
            else:
                rng = random.Random(zlib.crc32(rel.encode("utf-8")) ^ crop_seed)
                z = 1.0 / max(rng.uniform(lo, hi), 1e-6)
            row[FIELD] = [float(z)] * L
            vals.append(z)
        if a.shuffle is not None:
            import random as _r

            qs = sorted({row.get("query") for row in rows})
            vals_by_q = {}
            for row in rows:
                vals_by_q.setdefault(row.get("query"), row[FIELD][0])
            pool = [vals_by_q[q] for q in qs]
            _r.Random(a.shuffle).shuffle(pool)
            perm = dict(zip(qs, pool))
            for row in rows:
                row[FIELD + "_shuf"] = [float(perm[row.get("query")])] * L
            moved = sum(1 for q in qs if abs(perm[q] - vals_by_q[q]) > 1e-9)
            print(f"[{split}] 阴性对照 zoom_shuf:{moved}/{len(qs)} 个 query 换了值")
            if moved < 0.5 * len(qs):
                print(
                    f"[{split}] ⚠️ 这个 split 的变焦几乎是常数"
                    f"({len(set(round(v,3) for v in vals_by_q.values()))} 个不同取值)"
                    f",打乱对照在这里是退化的,不能用它下结论"
                )

        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(rows, fh)
        os.replace(tmp, path)
        if vals:
            import statistics

            print(
                f"[{split}] {len(rows)} rows, {miss} unmatched; zoom "
                f"min {min(vals):.2f} median {statistics.median(vals):.2f} "
                f"max {max(vals):.2f}, {len(set(round(v,3) for v in vals))} distinct"
            )
        from sl_agent.cache_fingerprint import refresh_output_fingerprint

        refresh_output_fingerprint(path, **{provenance_name: zoom_provenance})
    return 0


if __name__ == "__main__":
    sys.exit(main())
