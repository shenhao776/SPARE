#!/usr/bin/env python


import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS = ("database", "queries_train", "queries_val", "queries_test")
MANIFEST_NAME = ".layout_manifest.json"
DIGEST_FILE = os.path.join(ROOT, "configs", "layout_digests.json")
SCHEMA = 1


def sha256_file(path, block=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def _split_files(split_dir):
    out = []
    for dirpath, _dirs, files in os.walk(split_dir):
        for name in sorted(files):
            if name.startswith("."):
                continue
            path = os.path.join(dirpath, name)
            out.append((os.path.relpath(path, split_dir).replace(os.sep, "/"), path))
    return sorted(out)


def _roll_up(entries):

    h = hashlib.sha256()
    for rel, meta in sorted(entries.items()):
        h.update(f"{rel}\0{meta[0]}\0{meta[1]}\0".encode("utf-8"))
    return h.hexdigest()[:32]


def generation_params():

    import cv2

    from core.config import CFG
    from tools import generate_ill_queries as gen

    build = CFG.dataset.build
    crop = getattr(build, "crop", None)
    return {
        "build": {
            "min_dist": float(build.min_dist),
            "min_angle": float(build.min_angle),
            "intervals": {
                k: int(getattr(build.intervals, k)) for k in ("train", "val", "test")
            },
            "map_boundary_dist": float(getattr(build, "map_boundary_dist", 15.0)),
            "image_subdir": getattr(build, "image_subdir", "undistorted_images"),
            "full_frame_recordings": sorted(
                getattr(build, "full_frame_recordings", []) or []
            ),
            "crop_top_excl": float(getattr(crop, "top_excl", 0.0)) if crop else 0.0,
            "crop_bottom_excl": (
                float(getattr(crop, "bottom_excl", 0.3)) if crop else 0.3
            ),
        },
        "crop": {
            "seed": gen.CROP_SEED,
            "min_ratio": gen.MIN_CROP_RATIO,
            "max_ratio": gen.MAX_CROP_RATIO,
            "tilt_deg": gen.CROP_TILT_DEG,
            **({"warp_deg": gen.CROP_WARP_DEG} if gen.CROP_WARP_DEG > 0 else {}),
            **(
                {"warp_focal_ratio": gen.CROP_WARP_FOCAL_RATIO}
                if gen.CROP_WARP_DEG > 0
                else {}
            ),
            "centered": gen.CROP_CENTERED,
            "top_excl": gen.TOP_EXCL,
            "bottom_excl": gen.BOTTOM_EXCL,
            "pseudo_aug": gen.PSEUDO_AUG,
        },
        **(
            {"crop_test": {k: v for k, v in vars(getattr(build, "crop_test")).items()}}
            if getattr(build, "crop_test", None) is not None
            else {}
        ),
        "opencv": cv2.__version__,
    }


def build_manifest(scene, params):
    root = os.path.join(ROOT, "datasets", scene)
    splits = {}
    for split in SPLITS:
        d = os.path.join(root, split)
        if not os.path.isdir(d):
            continue
        entries = {}
        for rel, path in _split_files(d):
            entries[rel] = [os.path.getsize(path), sha256_file(path)]
        splits[split] = {
            "n": len(entries),
            "bytes": sum(v[0] for v in entries.values()),
            "digest": _roll_up(entries),
            "files": entries,
        }
    return {"schema": SCHEMA, "scene": scene, "params": params, "splits": splits}


def manifest_path(scene):
    return os.path.join(ROOT, "datasets", scene, MANIFEST_NAME)


def load_manifest(scene):
    try:
        with open(manifest_path(scene), "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def verify(scene, params, deep=False):

    manifest = load_manifest(scene)
    if manifest is None:
        return [f"{scene}: no {MANIFEST_NAME} (run --write)"], []
    hard, soft = [], []
    if manifest.get("schema") != SCHEMA:
        hard.append(f"{scene}: manifest schema {manifest.get('schema')} != {SCHEMA}")
    if manifest.get("params") != params:
        for key in sorted(set(manifest.get("params", {})) | set(params)):
            if manifest.get("params", {}).get(key) != params.get(key):
                hard.append(
                    f"{scene}: generation params differ in '{key}': manifest "
                    f"{manifest.get('params', {}).get(key)!r} vs current {params[key]!r}"
                )
    root = os.path.join(ROOT, "datasets", scene)
    for split, recorded in manifest.get("splits", {}).items():
        d = os.path.join(root, split)
        on_disk = dict(_split_files(d))
        want = recorded["files"]
        missing = sorted(set(want) - set(on_disk))
        extra = sorted(set(on_disk) - set(want))
        for rel in missing[:5]:
            hard.append(f"{scene}/{split}: missing {rel}")
        if len(missing) > 5:
            hard.append(f"{scene}/{split}: ... and {len(missing) - 5} more missing")
        for rel in extra[:5]:
            hard.append(f"{scene}/{split}: unexpected {rel}")
        if len(extra) > 5:
            hard.append(f"{scene}/{split}: ... and {len(extra) - 5} more unexpected")

        differing = 0
        first = None
        for rel in sorted(set(want) & set(on_disk)):
            size, digest = want[rel]
            path = on_disk[rel]
            if os.path.getsize(path) != size or (deep and sha256_file(path) != digest):
                differing += 1
                first = first or rel
        if differing:
            soft.append(
                f"{scene}/{split}: {differing}/{len(want)} files have the expected "
                f"name but different bytes (e.g. {first}) -- same crop, different "
                "encoder"
            )
    return hard, soft


def _scene_from_env():
    from core.config import TARGET_SCENE

    return TARGET_SCENE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", help="dataset selector; sets DATASET for config")
    ap.add_argument(
        "--write", action="store_true", help="hash what is on disk and stamp it"
    )
    ap.add_argument("--verify", action="store_true")
    ap.add_argument(
        "--deep", action="store_true", help="re-hash file contents on verify"
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="treat byte-level differences as failures too (default: "
        "only a differing file LIST or differing generation "
        "parameters fail; encoder-level byte differences are "
        "reported and tolerated)",
    )
    ap.add_argument(
        "--record",
        action="store_true",
        help="add this scene's roll-up digests to configs/layout_digests.json",
    )
    args = ap.parse_args()
    if args.dataset:
        os.environ["DATASET"] = args.dataset

    scene = _scene_from_env()
    params = generation_params()

    if args.write:
        manifest = build_manifest(scene, params)
        tmp = manifest_path(scene) + f".tmp.{os.getpid()}"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=1, sort_keys=True)
        os.replace(tmp, manifest_path(scene))
        for split, meta in sorted(manifest["splits"].items()):
            print(
                f"[manifest] {scene}/{split}: {meta['n']} files, "
                f"{meta['bytes'] / 1e9:.2f} GB, digest {meta['digest'][:16]}"
            )

    if args.verify:
        hard, soft = verify(scene, params, deep=args.deep)
        for p in hard:
            print(f"[layout] \u274c {p}")
        for p in soft:
            print(f"[layout] \u26a0\ufe0f  {p}")
        if hard or (soft and args.strict):
            raise SystemExit(1)
        if soft:
            print(
                f"[layout ok*] {scene}: same files and settings, some bytes differ "
                "-- tolerated (use --strict to fail instead)"
            )
        else:
            print(
                f"[layout ok] {scene}: matches {MANIFEST_NAME}"
                + (" (content-verified)" if args.deep else " (size-verified)")
            )

    if args.record:
        manifest = load_manifest(scene)
        if manifest is None:
            raise SystemExit(f"{scene}: no manifest to record; run --write first")
        try:
            with open(DIGEST_FILE, "r", encoding="utf-8") as fh:
                all_digests = json.load(fh)
        except (OSError, ValueError):
            all_digests = {}
        all_digests[scene] = {
            "params": manifest["params"],
            "splits": {
                s: {"n": m["n"], "bytes": m["bytes"], "digest": m["digest"]}
                for s, m in manifest["splits"].items()
            },
        }
        os.makedirs(os.path.dirname(DIGEST_FILE), exist_ok=True)
        with open(DIGEST_FILE, "w", encoding="utf-8") as fh:
            json.dump(all_digests, fh, indent=2, sort_keys=True)
        print(f"[manifest] recorded {scene} -> {os.path.relpath(DIGEST_FILE, ROOT)}")

    if not (args.write or args.verify or args.record):
        raise SystemExit("choose one of --write / --verify / --record")


if __name__ == "__main__":
    main()
