import fcntl
import glob
import hashlib
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIGEST_FILE = os.path.join(ROOT, "configs", "checkpoint_digests.json")
_HUB = os.path.expanduser("~/.cache/torch/hub/checkpoints")
_HF = os.path.expanduser("~/.cache/huggingface/hub")
_CACHE_FILE = os.path.join(ROOT, "assets", ".provenance_file_digests.json")

DIGEST_TARGETS = {
    "megaloc/model.safetensors": os.path.join(
        _HF, "models--gberton--MegaLoc", "snapshots", "*", "model.safetensors"
    ),
    "netvlad/vgg16": os.path.join(_HUB, "vgg16-397923af.pth"),
    "eigenplaces/resnet50": os.path.join(_HUB, "ResNet50_2048_eigenplaces.pth"),
    "salad/dino_salad.ckpt": os.path.join(_HUB, "dino_salad.ckpt"),
    "boq/dinov2_12288": os.path.join(_HUB, "dinov2_12288.pth"),
    "dinov2/vitb14_pretrain": os.path.join(_HUB, "dinov2_vitb14_pretrain.pth"),
    "lightglue/superpoint": os.path.join(_HUB, "superpoint_v1.pth"),
    "lightglue/matcher": os.path.join(_HUB, "superpoint_lightglue_v0-1_arxiv.pth"),
    "mixvpr/resnet50": os.path.join(
        ROOT, "assets", "MixVPR_4096", "resnet50_MixVPR_4096.ckpt"
    ),
    "cricavpr/CricaVPR": os.path.join(ROOT, "assets", "CricaVPR_10752", "CricaVPR.pth"),
    "supervlad/SuperVLAD": os.path.join(
        ROOT, "assets", "SuperVLAD_3072", "SuperVLAD.pth"
    ),
    "pairvpr/pairvpr-vitB": os.path.join(
        ROOT, "assets", "PairVPR_512", "pairvpr-vitB.pth"
    ),
}

_MODEL_IMPLICIT_KEYS = {
    "megaloc": ("megaloc/model.safetensors",),
    "netvlad": ("netvlad/vgg16",),
    "eigenplaces": ("eigenplaces/resnet50",),
    "salad": ("salad/dino_salad.ckpt",),
    "boq": ("boq/dinov2_12288",),
}
_MODEL_FIELD_KEYS = {
    ("mixvpr", "checkpoint"): "mixvpr/resnet50",
    ("cricavpr", "foundation_model_path"): "dinov2/vitb14_pretrain",
    ("cricavpr", "checkpoint"): "cricavpr/CricaVPR",
    ("supervlad", "foundation_model_path"): "dinov2/vitb14_pretrain",
    ("supervlad", "checkpoint"): "supervlad/SuperVLAD",
    ("pairvpr", "checkpoint"): "pairvpr/pairvpr-vitB",
}


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_reference_digests():
    try:
        with open(DIGEST_FILE, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return {}


def resolve_target_path(key):
    pattern = DIGEST_TARGETS[key]
    matches = (
        sorted(glob.glob(pattern))
        if "*" in pattern
        else ([pattern] if os.path.isfile(pattern) else [])
    )
    return matches[-1] if matches else None


def _sample_sha256(path, size, chunk=64 * 1024):

    digest = hashlib.sha256()
    offsets = sorted({0, max(0, size // 2 - chunk // 2), max(0, size - chunk)})
    with open(path, "rb") as handle:
        for offset in offsets:
            handle.seek(offset)
            digest.update(str(offset).encode("ascii"))
            digest.update(b"\0")
            digest.update(handle.read(chunk))
    return digest.hexdigest()


def verified_file_sha256(path, expected=None):

    path = os.path.abspath(path)
    stat = os.stat(path)
    identity = {
        "bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "ctime_ns": int(stat.st_ctime_ns),
        "sample_sha256": _sample_sha256(path, int(stat.st_size)),
    }
    os.makedirs(os.path.dirname(_CACHE_FILE), exist_ok=True)
    with open(_CACHE_FILE + ".lock", "a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            try:
                with open(_CACHE_FILE, encoding="utf-8") as handle:
                    cache = json.load(handle)
            except (OSError, ValueError):
                cache = {}
            entry = cache.get(path) or {}
            if any(entry.get(key) != value for key, value in identity.items()):
                entry = {**identity, "sha256": _sha256(path)}
                cache[path] = entry
                temporary = f"{_CACHE_FILE}.tmp.{os.getpid()}"
                with open(temporary, "w", encoding="utf-8") as handle:
                    json.dump(cache, handle, indent=2, sort_keys=True)
                os.replace(temporary, _CACHE_FILE)
            digest = entry["sha256"]
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    if expected and digest != expected:
        raise RuntimeError(
            f"checkpoint digest mismatch for {path}: expected {expected}, got {digest}"
        )
    return digest


def target_digest(key, references=None):
    path = resolve_target_path(key)
    if path is None:
        return "missing"
    references = load_reference_digests() if references is None else references
    expected = (references.get(key) or {}).get("sha256")
    return verified_file_sha256(path, expected=expected)


def local_feature_weight_signature():
    references = load_reference_digests()
    return {
        key: target_digest(key, references)
        for key in ("lightglue/superpoint", "lightglue/matcher")
    }


def model_weight_signature(selector, model):
    references = load_reference_digests()
    weights = {
        key: target_digest(key, references)
        for key in _MODEL_IMPLICIT_KEYS.get(selector, ())
    }
    for field in ("checkpoint", "foundation_model_path"):
        value = model.get(field)
        if not isinstance(value, str):
            continue
        path = value if os.path.isabs(value) else os.path.join(ROOT, value)
        reference_key = _MODEL_FIELD_KEYS.get((selector, field))
        expected = (references.get(reference_key) or {}).get("sha256")
        weights[field] = (
            verified_file_sha256(path, expected=expected)
            if os.path.isfile(path)
            else "missing"
        )
    return weights


def resolve_digests():
    found = {}
    for key in DIGEST_TARGETS:
        path = resolve_target_path(key)
        if path:
            found[key] = {
                "sha256": verified_file_sha256(path),
                "bytes": os.path.getsize(path),
            }
    return found


def compare_digests(current, recorded):
    mismatched = [
        f"{key}: recorded {value['sha256'][:12]} but this host has "
        f"{current[key]['sha256'][:12]}"
        for key, value in recorded.items()
        if key in current and current[key]["sha256"] != value["sha256"]
    ]
    missing = sorted(set(recorded) - set(current))
    return mismatched, missing
