#!/usr/bin/env python


from __future__ import annotations

import json
import os
import tempfile
import time
from collections import OrderedDict

DEFAULT_THRESHOLDS = (5.0, 10.0, 15.0)
_LEGACY_DESCRIPTIVE_EXTRA_FIELDS = {"backbone", "official_checkpoint"}


def add_eval_args(parser):
    parser.add_argument(
        "--pool",
        default=None,
        help="assets/<model_id> whose cached Top-10 is re-ranked "
        "(default: the paper's main backbone). A re-ranker's job is to "
        "reorder a GIVEN pool, so two re-rankers are only comparable on "
        "the same one -- and a pool caps what any of them can reach.",
    )
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=None,
        help="deprecated single-radius mode; overrides --success-thresholds",
    )
    parser.add_argument(
        "--success-thresholds",
        default="5,10,15",
        help="comma-separated success radii in metres (default: 5,10,15)",
    )
    parser.add_argument(
        "--output-dir",
        default="assets/baseline_results",
        help="directory for atomic per-scene JSON results",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="recompute even when a matching current-protocol result exists",
    )


def parse_thresholds(args):
    if args.success_threshold is not None:
        values = [float(args.success_threshold)]
    else:
        try:
            values = [
                float(item.strip())
                for item in args.success_thresholds.split(",")
                if item.strip()
            ]
        except ValueError as exc:
            raise SystemExit(f"invalid --success-thresholds: {exc}") from exc
    if not values or any(value <= 0 for value in values):
        raise SystemExit("success thresholds must contain positive numbers")
    return tuple(OrderedDict.fromkeys(values))


def default_pool():

    from core.paper_config import load_paper_config

    config = load_paper_config()
    return config["model_ids"][config["main_backbone"]]


def scene_inputs(root, scene, n_queries, get_all_images_recursive, pool=None):
    pool = pool or default_pool()
    cache_path = os.path.join(
        root,
        "assets",
        pool,
        f"test_trajectories_{scene}_no_roi.json",
    )
    if not os.path.isfile(cache_path):
        raise FileNotFoundError(f"{scene}: no {pool} cache: {cache_path}")
    with open(cache_path, encoding="utf-8") as handle:
        rows = json.load(handle)

    q_dir = os.path.join(root, "datasets", scene, "queries_test")
    db_dir = os.path.join(root, "datasets", scene, "database")
    q_paths = {
        os.path.splitext(os.path.basename(path))[0]: path
        for path in get_all_images_recursive(q_dir)
    }
    db_paths = {
        os.path.splitext(os.path.basename(path))[0]: path
        for path in get_all_images_recursive(db_dir)
    }

    row_queries = {str(r.get("query")) for r in rows}
    missing = row_queries - set(q_paths)
    if missing:
        raise RuntimeError(
            f"{scene}: {pool} 池缓存与当前数据不一致"
            f"(缓存 {len(missing)}/{len(row_queries)} 个 query 不在 "
            f"queries_test 中)—— 数据换代后的陈旧池缓存。先经 table2 构建阶段"
            f"重建该池缓存(或整跑 reproduce 让其自然重建)再跑基线。"
        )

    grouped = OrderedDict()
    for row in rows:
        grouped.setdefault(row["query"], []).append(row)
    qnames = list(grouped)
    if n_queries:
        qnames = qnames[:n_queries]

    work = []
    needed_db = set()
    problems = []
    for query in qnames:
        candidates = sorted(grouped[query], key=lambda item: item.get("rank", 10**9))[
            :10
        ]
        query_path = q_paths.get(os.path.splitext(os.path.basename(query))[0])
        if query_path is None:
            problems.append(f"missing query image {query}")
            continue
        resolved = []
        for candidate in candidates:
            stem = os.path.splitext(os.path.basename(candidate["candidate_name"]))[0]
            path = db_paths.get(stem)
            if path is None:
                problems.append(
                    f"missing database image {candidate['candidate_name']} "
                    f"for query {query}"
                )
                continue
            resolved.append((candidate, stem))
            needed_db.add(stem)
        if len(resolved) != len(candidates):
            continue
        if not resolved:
            problems.append(f"empty Top-10 pool for query {query}")
            continue
        work.append((query, query_path, resolved))

    if problems or len(work) != len(qnames):
        examples = "; ".join(problems[:5])
        raise RuntimeError(
            f"{scene}: refusing a biased denominator: valid={len(work)}/"
            f"{len(qnames)}; {examples}"
        )

    selected_db_paths = {stem: db_paths[stem] for stem in sorted(needed_db)}
    return cache_path, work, selected_db_paths


def evaluate_scenes(
    root,
    args,
    method,
    thresholds,
    checkpoint_paths,
    get_all_images_recursive,
    extract,
    score_candidates,
    metadata_extra=None,
    per_scene_setup=None,
):

    pool = getattr(args, "pool", None) or default_pool()
    for scene in [item.strip() for item in args.scenes.split(",") if item.strip()]:
        cache_path, work, db_paths = scene_inputs(
            root, scene, args.n_queries, get_all_images_recursive, pool=pool
        )
        extra = dict(metadata_extra or {})
        extra["n_queries_limit"] = int(args.n_queries)
        metadata = result_metadata(
            method, scene, cache_path, checkpoint_paths, extra, pool=pool
        )
        out_path = result_path(root, args.output_dir, method, scene, pool=pool)
        cached = None if args.force else matching_result(out_path, metadata, thresholds)
        if cached is not None:
            print(f"[resume] {scene}: matching result already exists")
            print_result(cached, out_path)
            continue

        db_cache_cap = int(os.environ.get("RERANK_DB_CACHE_CAP", "2000"))
        lazy = len(db_paths) > db_cache_cap
        if lazy:
            from collections import OrderedDict

            db_features = OrderedDict()

            def db_get(stem):
                if stem in db_features:
                    db_features.move_to_end(stem)
                    return db_features[stem]
                value = extract(db_paths[stem])
                db_features[stem] = value
                if len(db_features) > db_cache_cap:
                    db_features.popitem(last=False)
                return value

            print(
                f"[{scene}] {len(db_paths)} candidate db frames "
                f"(lazy LRU cache of {db_cache_cap})"
            )
        else:
            db_features = {}
            for stem, path in db_paths.items():
                db_features[stem] = extract(path)
                if len(db_features) % 250 == 0:
                    print(
                        f"  [{scene}] database features "
                        f"{len(db_features)}/{len(db_paths)}",
                        flush=True,
                    )
            print(f"[{scene}] extracted {len(db_features)} candidate db frames")

            def db_get(stem):
                return db_features[stem]

        if per_scene_setup is not None:
            if lazy:
                for stem in list(db_paths)[:50]:
                    db_get(stem)
            per_scene_setup(scene, work, db_features, extract)

        raw_errors, reranked_errors, selected_ranks = [], [], []
        started = time.time()
        for query_index, (query, query_path, resolved) in enumerate(work):
            query_feature = extract(query_path)
            candidates = [candidate for candidate, _ in resolved]
            candidate_features = [db_get(stem) for _, stem in resolved]
            scores = score_candidates(query_feature, candidate_features)
            if hasattr(scores, "detach"):
                scores = scores.detach().cpu().reshape(-1).tolist()
            else:
                scores = list(scores)
            if len(scores) != len(candidates):
                raise RuntimeError(
                    f"{scene}/{query}: scorer returned {len(scores)} values "
                    f"for {len(candidates)} candidates"
                )
            numeric_scores = [float(score) for score in scores]
            best_index = max(range(len(numeric_scores)), key=numeric_scores.__getitem__)
            best = candidates[best_index]
            raw_errors.append(float(candidates[0].get("error", -1)))
            reranked_errors.append(float(best.get("error", -1)))
            selected_ranks.append(int(best.get("rank", -1)))
            if query_index % 50 == 0:
                print(
                    f"  [{scene}] {query_index}/{len(work)} "
                    f"{time.time() - started:.0f}s",
                    flush=True,
                )
        result = save_result(
            out_path,
            metadata,
            thresholds,
            raw_errors,
            reranked_errors,
            selected_ranks,
            time.time() - started,
        )
        print_result(result, out_path)


POOL_IRRELEVANT_FIELDS = ("vprp_sim", "vprp_shuf")


def pool_identity(cache_path):

    import hashlib

    from core.checkpoint_provenance import verified_file_sha256

    with open(cache_path, "rb") as handle:
        raw = handle.read()
    if not any(f'"{key}"'.encode() in raw for key in POOL_IRRELEVANT_FIELDS):
        return {"size": len(raw), "sha256": verified_file_sha256(cache_path)}
    rows = json.loads(raw)
    for row in rows:
        for key in POOL_IRRELEVANT_FIELDS:
            row.pop(key, None)
    stripped = json.dumps(rows).encode()
    return {"size": len(stripped), "sha256": hashlib.sha256(stripped).hexdigest()}


def result_metadata(
    method, scene, cache_path, checkpoint_paths=(), extra=None, pool=None
):
    from core.checkpoint_provenance import verified_file_sha256

    def content_identity(path):
        info = os.stat(path)
        return {
            "size": info.st_size,
            "sha256": verified_file_sha256(path),
        }

    return {
        "method": method,
        "scene": scene,
        "pool": f"{pool or default_pool()} Top-10",
        "cache": pool_identity(cache_path),
        "checkpoints": [content_identity(path) for path in checkpoint_paths],
        "extra": extra or {},
    }


def result_path(root, output_dir, method, scene, pool=None):

    if not os.path.isabs(output_dir):
        output_dir = os.path.join(root, output_dir)
    safe_method = "".join(
        char.lower() if char.isalnum() else "_" for char in method
    ).strip("_")
    pool = pool or default_pool()
    return os.path.join(output_dir, f"{safe_method}_{scene}__pool-{pool}.json")


def _semantic_metadata(metadata):

    def identity(item):
        projected = {"size": item.get("size")}
        if item.get("sha256"):
            projected["sha256"] = item["sha256"]
        return projected

    extra = {
        key: value
        for key, value in (metadata.get("extra") or {}).items()
        if key not in _LEGACY_DESCRIPTIVE_EXTRA_FIELDS
    }
    return {
        "method": metadata.get("method"),
        "scene": metadata.get("scene"),
        "pool": metadata.get("pool"),
        "cache": identity(metadata.get("cache") or {}),
        "checkpoints": [identity(item) for item in (metadata.get("checkpoints") or [])],
        "extra": extra,
    }


def _metadata_compatible(previous, current):
    old = _semantic_metadata(previous)
    new = _semantic_metadata(current)
    if old["method"] != new["method"] or old["scene"] != new["scene"]:
        return False
    if old["pool"] != new["pool"] or old["extra"] != new["extra"]:
        return False
    if len(old["checkpoints"]) != len(new["checkpoints"]):
        return False

    def same_identity(left, right):
        if left.get("size") != right.get("size"):
            return False
        return not left.get("sha256") or left["sha256"] == right.get("sha256")

    return same_identity(old["cache"], new["cache"]) and all(
        same_identity(left, right)
        for left, right in zip(old["checkpoints"], new["checkpoints"])
    )


def _valid_error_payload(result):
    try:
        count = int(result.get("n_queries", 0))
        raw = result["raw_errors_m"]
        reranked = result["reranked_errors_m"]
        ranks = result["selected_ranks"]
    except (KeyError, TypeError, ValueError):
        return False
    return count > 0 and len(raw) == len(reranked) == len(ranks) == count


def matching_result(path, metadata, thresholds=DEFAULT_THRESHOLDS):
    try:
        with open(path, encoding="utf-8") as handle:
            result = json.load(handle)
    except (OSError, ValueError):
        return None
    if not _valid_error_payload(result):
        return None
    if not _metadata_compatible(result.get("metadata") or {}, metadata):
        return None

    normalized_thresholds = tuple(float(value) for value in thresholds)
    if (
        result.get("metadata") != metadata
        or tuple(result.get("thresholds_m") or ()) != normalized_thresholds
    ):
        result = save_result(
            path,
            metadata,
            normalized_thresholds,
            result["raw_errors_m"],
            result["reranked_errors_m"],
            result["selected_ranks"],
            result.get("elapsed_s", 0.0),
        )
        print(
            f"[cache migrated] {metadata['scene']}: refreshed result metadata "
            "and threshold summaries without model evaluation",
            flush=True,
        )
    return result


def _assert_sane(scene, raw_errors, raw_recall):

    n_bad = sum(1 for e in raw_errors if e is None or e < 0)
    if n_bad == len(raw_errors):
        raise RuntimeError(
            f"{scene}: 全部 {n_bad} 个 query 的 raw 误差都是哨兵值 —— "
            f"候选池缺 'error' 字段(常见于跨机搬运时只抽了部分字段)。"
            f"结果无效,不写盘。"
        )
    if all(v == 0.0 for v in raw_recall.values()):
        raise RuntimeError(
            f"{scene}: raw recall 全零 —— raw 是检索器的固有表现,不可能为 0。"
            f"多半是候选池与当前数据不匹配。结果无效,不写盘。"
        )


def _recall(errors, threshold):
    return 100.0 * sum(0 <= value <= threshold for value in errors) / len(errors)


def save_result(
    path,
    metadata,
    thresholds,
    raw_errors,
    reranked_errors,
    selected_ranks,
    elapsed_s,
):
    if not raw_errors or len(raw_errors) != len(reranked_errors):
        raise RuntimeError("result error arrays are empty or have inconsistent lengths")
    result = {
        "metadata": metadata,
        "n_queries": len(raw_errors),
        "thresholds_m": list(thresholds),
        "raw_recall": {
            f"{threshold:g}": _recall(raw_errors, threshold) for threshold in thresholds
        },
        "reranked_recall": {
            f"{threshold:g}": _recall(reranked_errors, threshold)
            for threshold in thresholds
        },
        "top1_moved": sum(rank != 1 for rank in selected_ranks),
        "elapsed_s": float(elapsed_s),
        "raw_errors_m": raw_errors,
        "reranked_errors_m": reranked_errors,
        "selected_ranks": selected_ranks,
    }
    _assert_sane(result["metadata"]["scene"], raw_errors, result["raw_recall"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=os.path.basename(path) + ".",
        suffix=".tmp",
        dir=os.path.dirname(path),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return result


def print_result(result, path):
    raw = "/".join(
        f"{result['raw_recall'][f'{threshold:g}']:.2f}"
        for threshold in result["thresholds_m"]
    )
    reranked = "/".join(
        f"{result['reranked_recall'][f'{threshold:g}']:.2f}"
        for threshold in result["thresholds_m"]
    )
    meta = result["metadata"]
    pool_label = (meta.get("pool") or "NetVLAD").split("_")[0]
    print(
        f"[{meta['scene']}] n={result['n_queries']} "
        f"Raw({pool_label})={raw} {meta['method']}={reranked} "
        f"(top1 moved {result['top1_moved']}/{result['n_queries']})"
    )
    print(f"[result] {path}", flush=True)
