import hashlib
import os

import torch

from sl_agent.cache_fingerprint import file_sha256

from core.config import (
    FEATURE_HULL_MIN_INLIERS,
    FEATURE_MATCH_MIN_INLIERS,
    FEATURE_MATCH_MIN_RAW,
    GEOMETRY_KEYS,
    GEOMETRY_VERSION,
    RANSAC_REPROJ_THRESHOLD,
)

CACHE_PROTOCOL = 1
METRICS_LEN = 4 + len(GEOMETRY_KEYS)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cpu_features(value):

    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_features(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_features(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_features(item) for item in value)
    return value


def _asset_signature(path):
    return [int(os.path.getsize(path)), file_sha256(path)]


class OfflineQueryMatchCache:

    def __init__(self, scene, query_folder, query_path, cropper, device, enabled=True):
        env_enabled = os.getenv("OFFLINE_QUERY_CACHE", "True").lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        self.enabled = bool(enabled and env_enabled)
        self.cropper = cropper
        self.device = device
        self.query_path = os.path.abspath(query_path)
        query_rel = os.path.relpath(
            self.query_path, os.path.abspath(os.path.join("datasets", scene))
        )
        key = hashlib.sha1(query_rel.encode("utf-8")).hexdigest()[:16]
        stem = os.path.splitext(os.path.basename(query_path))[0]
        self.path = os.path.join(
            "assets",
            "offline_query_cache",
            scene,
            query_folder,
            f"{stem}_{key}.pt",
        )
        self.meta = {
            "protocol": CACHE_PROTOCOL,
            "query_rel": query_rel.replace(os.sep, "/"),
            "query_sha256": _sha256(self.query_path) if self.enabled else "",
            "superpoint_max_keypoints": 2048,
            "min_raw_matches": FEATURE_MATCH_MIN_RAW,
            "min_inliers": FEATURE_MATCH_MIN_INLIERS,
            "min_hull_inliers": FEATURE_HULL_MIN_INLIERS,
            "ransac_px": RANSAC_REPROJ_THRESHOLD,
        }
        self.query_features_cpu = None
        self.pairs = {}
        self.dirty = False
        self._new_pairs_since_flush = 0
        if self.enabled:
            self._load()

    def _read_payload(self):
        try:
            payload = torch.load(self.path, map_location="cpu", weights_only=False)
            if payload.get("meta") != self.meta:
                return None
            if not isinstance(payload.get("pairs", {}), dict):
                return None
            return payload
        except (OSError, ValueError, TypeError, AttributeError, RuntimeError):
            return None

    def _load(self):
        payload = self._read_payload()
        if payload is None:
            return
        self.query_features_cpu = payload.get("query_features")
        self.pairs = payload.get("pairs", {})

    def query_features(self, query_cv):
        if self.enabled and self.query_features_cpu:
            return self.cropper.unpack_offline_features(
                self.query_features_cpu, self.device
            )
        features = self.cropper.extract_image(query_cv)
        if self.enabled and features:
            self.query_features_cpu = _cpu_features(features)
            self.dirty = True
        return features

    def geometric_metrics(self, rel_path, asset_path, query_features, step_features):

        signature = _asset_signature(asset_path)
        cached = self.pairs.get(rel_path) if self.enabled else None
        if cached and cached.get("asset_signature") == signature:
            values = cached.get("metrics", [])
            if len(values) == METRICS_LEN and cached.get("geom_v") == GEOMETRY_VERSION:
                return (*values[:4], dict(zip(GEOMETRY_KEYS, values[4:])))

        kp_q, kp_s, matches = self.cropper.match_features(query_features, step_features)
        img_size = None
        size_t = (query_features or {}).get("image_size")
        if size_t is not None:
            try:
                img_size = tuple(float(v) for v in size_t[0].tolist())
            except Exception:
                img_size = None
        metrics = self.cropper.compute_geometric_metrics(
            kp_q, kp_s, matches, with_geometry=True, img_size=img_size
        )
        if self.enabled:
            self.pairs[rel_path] = {
                "asset_signature": signature,
                "geom_v": GEOMETRY_VERSION,
                "metrics": list(metrics[:4]) + [metrics[4][k] for k in GEOMETRY_KEYS],
            }
            self.dirty = True
            self._new_pairs_since_flush += 1
            if self._new_pairs_since_flush >= 32:
                self.flush()
        return metrics

    def flush(self):
        if not self.enabled or not self.dirty:
            return
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        lock_path = self.path + ".lock"
        with open(lock_path, "a+b") as lock_handle:
            try:
                import fcntl

                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            except ImportError:
                pass
            current = self._read_payload()
            if current is not None:
                merged_pairs = dict(current.get("pairs", {}))
                merged_pairs.update(self.pairs)
                self.pairs = merged_pairs
                if self.query_features_cpu is None:
                    self.query_features_cpu = current.get("query_features")
            payload = {
                "meta": self.meta,
                "query_features": self.query_features_cpu,
                "pairs": self.pairs,
            }
            tmp = f"{self.path}.tmp.{os.getpid()}"
            torch.save(payload, tmp)
            os.replace(tmp, self.path)
        self.dirty = False
        self._new_pairs_since_flush = 0
