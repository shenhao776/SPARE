#!/usr/bin/env python


import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("DATASET", "robotcar")

from tools.rerank_eval_common import (
    add_eval_args,
    default_pool,
    matching_result,
    parse_thresholds,
    print_result,
    result_metadata,
    result_path,
    save_result,
    scene_inputs,
)

METHODS = {
    "inliers": "RoMa-GV-inliers",
    "certainty": "RoMa-GV-certainty",
}

DUTY_FILE = os.environ.get("ROMA_DUTY_FILE", "/tmp/spare_roma_duty")


def _current_duty(default):
    try:
        with open(DUTY_FILE, encoding="utf-8") as fh:
            value = float(fh.read().strip())
        if 0.0 < value <= 1.0:
            return value
    except (OSError, ValueError):
        pass
    return default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--n-queries", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--coarse-res", type=int, default=560)
    ap.add_argument("--upsample-res", type=int, default=864)
    ap.add_argument(
        "--no-upsample", action="store_true", help="skip RoMa's refinement upsampling"
    )
    ap.add_argument("--num-matches", type=int, default=2000)
    ap.add_argument("--ransac-px", type=float, default=5.0)
    ap.add_argument(
        "--duty-cycle",
        type=float,
        default=1.0,
        help="fraction of wall time the GPU may be busy (0.5 = sleep as long as "
        "each pair took); only a courtesy throttle on a shared desktop",
    )
    ap.add_argument(
        "--bench-json",
        default=None,
        help="timing-only mode: process the (limited) queries at full duty, write "
        "{ms_per_query, ms_per_pair, ...} of pure matching time to this path and do "
        "NOT write the re-ranking result JSONs (a limited run must never overwrite "
        "the full-set result under the same file name)",
    )
    ap.add_argument(
        "--dump-state",
        default=None,
        help="front-end swap test: for every query/candidate pair also reduce the "
        "RoMa correspondences to SPARE's geometry channels with the SAME function "
        "the SuperPoint+LightGlue builder uses (compute_geometric_metrics: RANSAC "
        "inliers, hull areas, scale ratio, hull shape, homography) and write "
        "{scene: {query: {rank: {...}}}} to this path. Never writes or resumes the "
        "re-ranking result JSONs (RoMa sampling is stochastic; the published "
        "baseline rows must not be overwritten by a re-run)",
    )
    ap.add_argument(
        "--shard",
        default=None,
        help="K/N: score only queries K, K+N, K+2N, ... of each scene and write "
        "<result>__shardKofN.json (RoMa Pitts30k is ~17 h on one GPU; shards can run "
        "on several machines from byte-identical copies of the cache and images)",
    )
    ap.add_argument(
        "--merge-shards",
        type=int,
        default=0,
        help="N: merge the N shard files of each scene into the normal result file "
        "(metadata must match exactly, query order is restored)",
    )
    add_eval_args(ap)
    a = ap.parse_args()
    if a.bench_json:
        a.duty_cycle = 1.0
    thresholds = parse_thresholds(a)

    torch.set_float32_matmul_precision("highest")
    from core.vpr_core import get_all_images_recursive
    from PIL import Image
    from romatch import roma_outdoor

    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    model = roma_outdoor(
        device=device,
        coarse_res=a.coarse_res,
        upsample_res=(a.upsample_res, a.upsample_res),
        upsample_preds=not a.no_upsample,
        use_custom_corr=False,
    )
    hub = os.path.join(torch.hub.get_dir(), "checkpoints")
    checkpoints = [
        os.path.join(hub, "roma_outdoor.pth"),
        os.path.join(hub, "dinov2_vitl14_pretrain.pth"),
    ]
    print(
        f"[roma] loaded roma_outdoor on {device}, coarse {a.coarse_res}, upsample {not a.no_upsample}"
    )
    extra = {
        "coarse_res": a.coarse_res,
        "upsample_res": a.upsample_res,
        "upsample": not a.no_upsample,
        "num_matches": a.num_matches,
        "ransac_px": a.ransac_px,
        "n_queries_limit": int(a.n_queries),
    }
    pool = a.pool or default_pool()

    size_cache = {}
    dump = {} if a.dump_state else None
    if dump is not None:
        from core.vpr_core import GEOMETRY_KEYS, DynamicCropProjector

    def image_size(path):
        if path not in size_cache:
            size_cache[path] = Image.open(path).size
        return size_cache[path]

    def pair_scores(query_path, cand_path):

        with torch.inference_mode():
            warp, certainty = model.match(query_path, cand_path, device=device)
            matches, cert = model.sample(warp, certainty, num=a.num_matches)
            cert_sum = float(cert.sum().item())
            wq, hq = image_size(query_path)
            wc, hc = image_size(cand_path)
            kq, kc = model.to_pixel_coordinates(matches, hq, wq, hc, wc)
        kq = kq.detach().cpu().numpy().astype(np.float32)
        kc = kc.detach().cpu().numpy().astype(np.float32)
        geom = None
        if dump is not None:
            kp_q = [cv2.KeyPoint(float(x), float(y), 1.0) for x, y in kq]
            kp_c = [cv2.KeyPoint(float(x), float(y), 1.0) for x, y in kc]
            pairs = [cv2.DMatch(i, i, 0.0) for i in range(len(kq))]
            vi, sr, kas, kaq, geo = DynamicCropProjector.compute_geometric_metrics(
                None, kp_q, kp_c, pairs, with_geometry=True, img_size=(wq, hq)
            )
            geom = {
                "inliers": int(vi),
                "scale_ratio": float(sr),
                "area_step": float(kas),
                "area_query": float(kaq),
                **{k: float(geo[k]) for k in GEOMETRY_KEYS},
                "certainty": cert_sum,
            }
        if len(kq) < 4:
            return 0.0, cert_sum, geom
        _, mask = cv2.findHomography(kc, kq, cv2.RANSAC, a.ransac_px)
        return (0.0 if mask is None else float(mask.sum())), cert_sum, geom

    for scene in [s.strip() for s in a.scenes.split(",") if s.strip()]:
        cache_path, work, _db_paths = scene_inputs(
            ROOT, scene, a.n_queries, get_all_images_recursive, pool=pool
        )
        metadata = {
            key: result_metadata(
                method, scene, cache_path, checkpoints, dict(extra), pool=pool
            )
            for key, method in METHODS.items()
        }
        out_paths = {
            key: result_path(ROOT, a.output_dir, method, scene, pool=pool)
            for key, method in METHODS.items()
        }

        def shard_path(path, k, n):
            return path[: -len(".json")] + f"__shard{k}of{n}.json"

        if a.merge_shards:
            n_sh = a.merge_shards
            expected = len(work)
            for key in METHODS:
                parts = []
                for k in range(n_sh):
                    with open(
                        shard_path(out_paths[key], k, n_sh), encoding="utf-8"
                    ) as fh:
                        part = json.load(fh)
                    meta_k = dict(part["metadata"])
                    extra_k = dict(meta_k.get("extra") or {})
                    if extra_k.pop("shard", None) != f"{k}/{n_sh}":
                        raise SystemExit(
                            f"{scene}/{key}: shard {k} carries a different shard tag"
                        )
                    meta_k["extra"] = extra_k
                    if meta_k != metadata[key]:
                        raise SystemExit(
                            f"{scene}/{key}: shard {k} metadata differs from this machine's "
                            "(cache, checkpoints or settings) -- not mergeable"
                        )
                    parts.append(part)
                if sum(p["n_queries"] for p in parts) != expected:
                    raise SystemExit(
                        f"{scene}/{key}: shards hold {sum(p['n_queries'] for p in parts)} of {expected} queries"
                    )
                raw_m, rer_m, sel_m = (
                    [None] * expected,
                    [None] * expected,
                    [None] * expected,
                )
                for k, part in enumerate(parts):
                    for j, i in enumerate(range(k, expected, n_sh)):
                        raw_m[i] = part["raw_errors_m"][j]
                        rer_m[i] = part["reranked_errors_m"][j]
                        sel_m[i] = part["selected_ranks"][j]
                result = save_result(
                    out_paths[key],
                    metadata[key],
                    thresholds,
                    raw_m,
                    rer_m,
                    sel_m,
                    sum(float(p["elapsed_s"]) for p in parts),
                )
                print_result(result, out_paths[key])
            continue

        if a.shard:
            k_sh, n_sh = (int(v) for v in a.shard.split("/"))
            work = work[k_sh::n_sh]
            for key in METHODS:
                metadata[key]["extra"]["shard"] = f"{k_sh}/{n_sh}"
            out_paths = {key: shard_path(out_paths[key], k_sh, n_sh) for key in METHODS}
            print(f"[shard] {scene}: {k_sh}/{n_sh} -> {len(work)} queries")

        if not a.force and dump is None:
            cached = {
                key: matching_result(out_paths[key], metadata[key], thresholds)
                for key in METHODS
            }
            if all(value is not None for value in cached.values()):
                for key in METHODS:
                    print(
                        f"[resume] {scene}/{METHODS[key]}: matching result already exists"
                    )
                    print_result(cached[key], out_paths[key])
                continue

        raw_errors = []
        reranked = {key: [] for key in METHODS}
        selected = {key: [] for key in METHODS}
        started = time.time()
        compute_s, n_pairs = 0.0, 0
        for qi, (query, query_path, resolved) in enumerate(work):
            cands = [candidate for candidate, _ in resolved]
            paths = [_db_paths[stem] for _, stem in resolved]
            scores = []
            for cand_path in paths:
                t_pair = time.time()
                scores.append(pair_scores(query_path, cand_path))
                busy = time.time() - t_pair
                compute_s += busy
                n_pairs += 1
                duty = 1.0 if a.bench_json else _current_duty(a.duty_cycle)
                if duty < 1.0:
                    time.sleep(busy * (1.0 - duty) / max(duty, 1e-3))
            if dump is not None:
                dump.setdefault(scene, {})[str(query)] = {
                    str(c.get("rank")): sc[2] for c, sc in zip(cands, scores)
                }
            raw_errors.append(float(cands[0].get("error", -1)))
            for idx, key in enumerate(METHODS):
                values = [s[idx] for s in scores]
                best = cands[max(range(len(values)), key=values.__getitem__)]
                reranked[key].append(float(best.get("error", -1)))
                selected[key].append(int(best.get("rank", -1)))
            if qi % 25 == 0:
                print(
                    f"  [{scene}] {qi}/{len(work)} {time.time() - started:.0f}s",
                    flush=True,
                )
        elapsed = time.time() - started
        if dump is not None:
            out = {"metadata": {}, "pairs": {}}
            if os.path.isfile(a.dump_state):
                with open(a.dump_state, encoding="utf-8") as fh:
                    out = json.load(fh)
            out["metadata"][scene] = metadata["inliers"]
            out["pairs"][scene] = dump[scene]
            os.makedirs(os.path.dirname(os.path.abspath(a.dump_state)), exist_ok=True)
            with open(a.dump_state, "w", encoding="utf-8") as fh:
                json.dump(out, fh)
            for key in METHODS:
                rec = {
                    t: 100.0 * sum(0 <= e <= t for e in reranked[key]) / len(work)
                    for t in thresholds
                }
                print(f"[dump] {scene} {METHODS[key]} (this run, not saved): {rec}")
            print(f"[dump] {scene}: {len(dump[scene])} queries -> {a.dump_state}")
            continue
        if a.bench_json:
            n_q = len(work)
            bench = {
                "scene": scene,
                "n_queries": n_q,
                "n_pairs": n_pairs,
                "pairs_per_query": n_pairs / max(n_q, 1),
                "ms_per_pair": 1000.0 * compute_s / max(n_pairs, 1),
                "ms_per_query": 1000.0 * compute_s / max(n_q, 1),
                "wall_s": elapsed,
                "device": str(device),
                "gpu": (
                    torch.cuda.get_device_name(0)
                    if torch.cuda.is_available()
                    else "cpu"
                ),
                "coarse_res": a.coarse_res,
                "upsample": not a.no_upsample,
                "num_matches": a.num_matches,
            }
            os.makedirs(os.path.dirname(os.path.abspath(a.bench_json)), exist_ok=True)
            with open(a.bench_json, "w", encoding="utf-8") as fh:
                json.dump(bench, fh, indent=1)
            print(
                f"[bench] {scene}: {bench['ms_per_query']:.0f} ms/query over {n_q} queries -> {a.bench_json}"
            )
            continue
        for key in METHODS:
            result = save_result(
                out_paths[key],
                metadata[key],
                thresholds,
                raw_errors,
                reranked[key],
                selected[key],
                elapsed,
            )
            print_result(result, out_paths[key])


if __name__ == "__main__":
    main()
