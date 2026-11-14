import collections
import hashlib
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHECK_ONLY = os.getenv("BUILD_RESUME_CHECK") == "1"
CHECK_REBUILD_EXIT = 10
FORCE_REBUILD = os.getenv("FORCE_REBUILD_CACHE") == "1"

_FILE_DIGESTS = {}


def file_sha256(path, use_cache=True):

    path = os.path.abspath(path)
    stat = os.stat(path)
    key = (path, int(stat.st_size), int(stat.st_mtime_ns), int(stat.st_ctime_ns))
    if not use_cache or key not in _FILE_DIGESTS:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(block)
        value = digest.hexdigest()
        if use_cache:
            _FILE_DIGESTS[key] = value
        return value
    return _FILE_DIGESTS[key]


def _json_digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode(
            "utf-8"
        )
    ).hexdigest()


_RETRIEVER_KEY_FIELDS = ("transform", "weight_files")


def _model_source_signature():

    from core.checkpoint_provenance import model_weight_signature
    from core.config import CFG

    selector = CFG.model.selector
    model = dict(CFG.model_dict)
    return {
        "transform": model.get("transform"),
        "weight_files": model_weight_signature(selector, model),
    }


def validate_sequence_asset(payload, expected_rel_path, scene):

    from core.config import SEQUENCE_ASSET_FORMAT

    expected_rel_path = str(expected_rel_path).replace("\\", "/")
    if not isinstance(payload, dict):
        return "payload is not a dict"
    if payload.get("format") != SEQUENCE_ASSET_FORMAT:
        return (
            f"format={payload.get('format')!r}, expected " f"{SEQUENCE_ASSET_FORMAT!r}"
        )
    stored = str(payload.get("source_rel_path", "")).replace("\\", "/")
    if stored != expected_rel_path:
        return f"source_rel_path={stored!r}, expected {expected_rel_path!r}"
    paths = [
        str(value).replace("\\", "/") for value in (payload.get("rel_seq_paths") or [])
    ]
    if paths != [expected_rel_path]:
        return f"rel_seq_paths={paths!r}, expected one matching frame"
    feats = payload.get("lightglue_feats") or []
    if len(feats) != 1:
        return f"contains {len(feats)} feature sets, expected 1"
    return None


def state_build_settings(split):

    from core.config import (
        DATA_ABLATION_RANDOM_WINDOW,
        FEATURE_HULL_MIN_INLIERS,
        FEATURE_MATCH_MIN_INLIERS,
        FEATURE_MATCH_MIN_RAW,
        GEOMETRY_VERSION,
        MIN_CANDIDATE_DIST,
        RANSAC_REPROJ_THRESHOLD,
        SPATIAL_STEPS,
        STEP_BUDGET,
        STEP_FEATURE_TAU,
        STEP_MAX_DIST,
        TOP_K,
        VPR_MODEL_ID,
    )

    return {
        "temporal_protocol": 2,
        "split": split,
        "model_id": VPR_MODEL_ID,
        "top_k": TOP_K,
        "min_candidate_dist": MIN_CANDIDATE_DIST,
        "random_window": DATA_ABLATION_RANDOM_WINDOW,
        "no_roi": True,
        "min_raw_matches": FEATURE_MATCH_MIN_RAW,
        "min_inliers": FEATURE_MATCH_MIN_INLIERS,
        "min_hull_inliers": FEATURE_HULL_MIN_INLIERS,
        "ransac_px": RANSAC_REPROJ_THRESHOLD,
        "geometry_version": GEOMETRY_VERSION,
        "expected_steps": sum(SPATIAL_STEPS) + 1,
        "spatial_steps": list(SPATIAL_STEPS),
        "step_feature_tau": STEP_FEATURE_TAU,
        "step_max_dist": STEP_MAX_DIST,
        "step_budget": STEP_BUDGET,
        "retriever": _model_source_signature(),
    }


def meta_path(output_json):
    return output_json + ".meta.json"


STRUCTURE_SCHEMA = 1


def build_fingerprint(query_paths, settings):

    canonical_settings = json.loads(json.dumps(settings, sort_keys=True, default=str))
    identity = {
        "n_queries": len(query_paths),
        "settings": canonical_settings,
    }
    cache_key = _json_digest(identity)
    return {
        **identity,
        "cache_key": cache_key,
        "structure_schema": STRUCTURE_SCHEMA,
        "semantic_digest": cache_key,
    }


def read_fingerprint(output_json):
    try:
        with open(meta_path(output_json), "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def write_fingerprint(output_json, fingerprint):

    path = meta_path(output_json)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(
            {
                **fingerprint,
                "complete": True,
            },
            handle,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
    os.replace(tmp, path)


def refresh_output_fingerprint(output_json, **provenance):

    stamp = read_fingerprint(output_json)
    if not stamp or not stamp.get("complete") or not stamp.get("semantic_digest"):
        raise ValueError(
            f"cannot update provenance for unverified state cache: {output_json}"
        )
    if provenance:
        injectors = dict(stamp.get("injectors") or {})
        injectors.update(provenance)
        stamp["injectors"] = injectors
    path = meta_path(output_json)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(stamp, handle, indent=2, ensure_ascii=False, sort_keys=True)
    os.replace(tmp, path)


INJECTOR_FIELDS = {
    "dino": ("dino_sim",),
    "step_meta": ("step_offset_m", "step_turn_deg", "step_change", "step_valid"),
    "zoom": ("zoom",),
    "zoom_shuf": ("zoom_shuf",),
    "vprpatch": ("vprp_sim", "vprp_shuf"),
}


def injected_shape_error(output_json, name, expected_steps, sample=200):

    fields = INJECTOR_FIELDS.get(name)
    if not fields or not expected_steps:
        return None
    try:
        with open(output_json, "r", encoding="utf-8") as handle:
            rows = json.load(handle)
    except (OSError, ValueError) as exc:
        return f"cache unreadable: {exc}"
    if not isinstance(rows, list) or not rows:
        return "cache has no rows"
    step = max(1, len(rows) // sample)
    for row in rows[::step][:sample]:
        if not isinstance(row, dict):
            return "cache rows are not objects"
        for field in fields:
            seq = row.get(field)
            if seq is None:
                return f"'{field}' is missing"
            if not isinstance(seq, list):
                return f"'{field}' is not a sequence"
            if len(seq) != expected_steps:
                return (
                    f"'{field}' has {len(seq)} steps, expected {expected_steps} "
                    "-- the producer did not walk a full window"
                )
    return None


def injector_current(output_json, name, provenance, expected_steps=None):

    stamp = read_fingerprint(output_json) or {}
    if not stamp.get("complete"):
        return False
    injectors = dict(stamp.get("injectors") or {})
    old = injectors.get(name)
    legacy_fields = list(stamp.get("legacy_present_fields") or [])
    inputs_match = old == provenance or (name in legacy_fields and old is None)
    if not inputs_match and isinstance(old, dict):
        simplified = {
            key: value
            for key, value in old.items()
            if not key.endswith("ast_sha256") and key != "content_version"
        }
        inputs_match = simplified == {
            key: value for key, value in provenance.items() if key != "content_version"
        }
    if not inputs_match:
        return False
    if injected_shape_error(output_json, name, expected_steps):
        return False
    if old != provenance or name in legacy_fields:
        injectors[name] = provenance
        stamp["injectors"] = injectors
        if name in legacy_fields:
            stamp["legacy_present_fields"] = [
                field for field in legacy_fields if field != name
            ]
        _write_meta(meta_path(output_json), stamp)
    return True


TEMPORAL_FIELDS = (
    "inliers",
    "scores",
    "step_errors",
    "area_step",
    "area_query",
    "scale_ratio",
)


def clone_rows_stat(rows_of_query):

    if len(rows_of_query) < 2:
        return 0
    seqs = {tuple(row.get("step_errors") or ()) for row in rows_of_query}
    if len(seqs) != 1:
        return 0
    names = {row.get("candidate_name") for row in rows_of_query}
    return 1 if len(names) > 1 else 0


def cache_stats(output_json):
    try:
        with open(output_json, "r", encoding="utf-8") as handle:
            rows = json.load(handle)
    except (OSError, ValueError):
        return None
    if not isinstance(rows, list):
        return None
    per_query = collections.Counter(row.get("query") for row in rows)
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[row.get("query")].append(row)
    return {
        "rows": len(rows),
        "queries": set(per_query),
        "rows_per_query": sorted(set(per_query.values())),
        "step_lengths": sorted(
            {len(row.get(field) or []) for row in rows for field in TEMPORAL_FIELDS}
        ),
        "degenerate": sum(
            1
            for row in rows
            if len(row.get("step_errors") or []) > 1
            and len(set(row["step_errors"])) == 1
        ),
        "clone_groups": sum(clone_rows_stat(group) for group in grouped.values()),
        "groups": len(grouped),
    }


def structural_error(stats, settings):
    if not stats or not stats["rows"]:
        return "cache has no rows"
    counts = stats["rows_per_query"]
    top_k = settings.get("top_k")
    if len(counts) > 1:
        return (
            "rows per query is not uniform ("
            + ", ".join(str(count) for count in counts)
            + ") -- a shard merge duplicated rows"
        )
    if top_k and counts and counts[0] > top_k:
        return (
            f"cache has {counts[0]} rows per query but top_k={top_k} "
            "-- a shard merge duplicated rows"
        )
    expected = settings.get("expected_steps")
    if expected:
        wrong = [length for length in stats["step_lengths"] if length != expected]
        if wrong:
            return (
                f"rows are not {expected}-step trajectories (found lengths "
                + ", ".join(str(length) for length in wrong)
                + ")"
            )
    limit = max(5, int(0.01 * stats["rows"]))
    if stats["degenerate"] > limit:
        return (
            f"{stats['degenerate']}/{stats['rows']} rows have a constant "
            "step_errors trajectory -- the window collapsed to one repeated frame"
        )
    clones = stats.get("clone_groups") or 0
    if clones:
        return (
            f"{clones}/{stats.get('groups', '?')} queries have IDENTICAL "
            "step_errors on every candidate -- the rows were written from a "
            "single candidate's window; rebuild this cache"
        )
    return None


BASELINE_IRRELEVANT_INJECTORS = ("vprpatch",)


def trajectory_fingerprint(
    scene,
    model_id="MegaLoc_8448",
    protocol=2,
    data_suffix="_no_roi",
):

    cache = os.path.join(
        ROOT,
        "assets",
        model_id,
        f"test_trajectories_{scene}{data_suffix}.json",
    )
    meta = meta_path(cache)
    if not os.path.isfile(cache) or not os.path.isfile(meta):
        raise FileNotFoundError(f"missing test cache or sidecar for {scene}: {cache}")
    stamp = read_fingerprint(cache) or {}
    cache_key = stamp.get("cache_key") or stamp.get("semantic_digest")
    if not stamp.get("complete") or not cache_key:
        raise ValueError(f"unverified legacy test cache: {cache}; rebuild it")
    injectors = {
        name: value
        for name, value in (stamp.get("injectors") or {}).items()
        if name not in BASELINE_IRRELEVANT_INJECTORS
    }
    digest = hashlib.sha256()
    digest.update(str(protocol).encode("ascii"))
    digest.update(cache_key.encode("ascii"))
    digest.update(_json_digest(injectors).encode("ascii"))
    return digest.hexdigest()


def _legacy_settings_projection(settings):

    value = dict(settings or {})
    for key in (
        "content_version",
        "layout_query_digest",
        "producer_ast_sha256",
    ):
        value.pop(key, None)

    retriever = value.get("retriever") or {}
    value["retriever"] = {
        key: retriever.get(key)
        for key in ("selector", "model_id", "weight_files", "hub_repo", "hub_entry")
    }
    sequence = value.get("sequence_assets") or {}
    weights = sequence.get("local_feature_weights")
    if weights is None:
        weights = (sequence.get("local_features") or {}).get("weights")
    value["sequence_assets"] = {
        "format": sequence.get("format"),
        "scene": sequence.get("scene"),
        "layout_database_digest": sequence.get("layout_database_digest"),
        "local_feature_weights": weights,
    }
    return value


def content_settings_compatible(old, new):

    return _legacy_settings_projection(old) == _legacy_settings_projection(new)


def _write_meta(path, payload):
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
    os.replace(tmp, path)


def _legacy_present_fields(output_json):

    try:
        with open(output_json, "r", encoding="utf-8") as handle:
            prefix = handle.read(2 * 1024 * 1024)
    except OSError:
        return []
    markers = {
        "step_meta": '"step_offset_m"',
        "dino": '"dino_sim"',
        "zoom": '"zoom"',
        "zoom_shuf": '"zoom_shuf"',
    }
    return [name for name, marker in markers.items() if marker in prefix]


def _structure_gate(output_json, stamp, settings):

    if stamp.get("structure_schema") == STRUCTURE_SCHEMA:
        return None
    stats = cache_stats(output_json)
    if stats is None:
        return "cache is unreadable"
    broken = structural_error(stats, settings or {})
    if broken:
        return f"structurally broken: {broken}"
    stamp = {**stamp, "structure_schema": STRUCTURE_SCHEMA}
    _write_meta(meta_path(output_json), stamp)
    return None


def settings_still_match(old_settings, new_settings):

    for key, value in new_settings.items():
        recorded = old_settings.get(key)
        if (
            key == "retriever"
            and isinstance(recorded, dict)
            and isinstance(value, dict)
        ):
            if any(recorded.get(field) != value.get(field) for field in value):
                return False
            continue
        if recorded != value:
            return False
    return True


def _adopt(output_json, migrated, settings):

    migrated = {k: v for k, v in migrated.items() if k != "structure_schema"}
    _write_meta(meta_path(output_json), migrated)
    return _structure_gate(output_json, migrated, settings)


def resume_reason(output_json, fingerprint, query_paths=None):
    if FORCE_REBUILD:
        return "FORCE_REBUILD_CACHE=1"
    if not os.path.exists(output_json):
        return "no cache file"
    old = read_fingerprint(output_json)
    if old is None:
        return "没有戳"
    if not old.get("complete"):
        return "sidecar 缺 complete 标记"
    if old.get("cache_key") == fingerprint.get("cache_key"):
        return _structure_gate(output_json, old, fingerprint.get("settings"))

    old_settings = old.get("settings") or {}
    new_settings = fingerprint.get("settings") or {}
    if (
        old.get("complete") is True
        and old.get("cache_key")
        and old.get("n_queries") == fingerprint.get("n_queries")
        and new_settings
        and settings_still_match(old_settings, new_settings)
    ):
        migrated = {
            **fingerprint,
            "complete": True,
            "injectors": old.get("injectors") or {},
            "legacy_present_fields": (
                old.get("legacy_present_fields") or _legacy_present_fields(output_json)
            ),
            "migrated_from": "preprocessing-digest-key",
        }
        return _adopt(output_json, migrated, new_settings)

    if (
        not old.get("cache_key")
        and old.get("semantic_digest")
        and old.get("n_queries") == fingerprint.get("n_queries")
        and content_settings_compatible(
            old.get("settings"), fingerprint.get("settings")
        )
    ):
        migrated = {
            **fingerprint,
            "complete": True,
            "injectors": old.get("injectors") or {},
            "migrated_from": "ast-output-sha256",
        }
        return _adopt(output_json, migrated, fingerprint.get("settings"))

    old_settings = old.get("settings") or {}
    new_settings = fingerprint.get("settings") or {}
    if (
        not old.get("cache_key")
        and old.get("n_queries") == fingerprint.get("n_queries")
        and old_settings
        and all(new_settings.get(key) == value for key, value in old_settings.items())
    ):
        migrated = {
            **fingerprint,
            "complete": True,
            "injectors": old.get("injectors") or {},
            "legacy_present_fields": _legacy_present_fields(output_json),
            "migrated_from": "legacy-v1-minimal",
        }
        return _adopt(output_json, migrated, new_settings)

    if old.get("cache_key") != fingerprint.get("cache_key"):
        changed = []
        for key in ("version", "n_queries"):
            if old.get(key) != fingerprint.get(key):
                changed.append(f"{key}: {old.get(key)!r} -> {fingerprint.get(key)!r}")
        old_settings = old.get("settings") or {}
        new_settings = fingerprint.get("settings") or {}
        for key in sorted(set(old_settings) | set(new_settings)):
            if old_settings.get(key) != new_settings.get(key):
                changed.append(
                    f"settings.{key}: {old_settings.get(key)!r} -> {new_settings.get(key)!r}"
                )
        return "content key changed: " + "; ".join(changed[:6])
    return None
