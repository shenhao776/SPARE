#!/usr/bin/env python


import argparse
import contextlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.checkpoint_provenance import (
    DIGEST_FILE,
    compare_digests,
    resolve_digests,
)

FILES = [
    {
        "relative": "MixVPR_4096/resnet50_MixVPR_4096.ckpt",
        "min_bytes": 40_000_000,
        "gdrive_id": "1vuz3PvnR7vxnDDLQrdHJaOA04SQrtk5L",
        "vendored": (
            "3rdparty/AnyLoc/MixVPR/"
            "resnet50_MixVPR_4096_channels(1024)_rows(4) (1).ckpt"
        ),
    },
    {
        "relative": "CricaVPR_10752/dinov2_vitb14_pretrain.pth",
        "min_bytes": 300_000_000,
        "url": (
            "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/"
            "dinov2_vitb14_pretrain.pth"
        ),
    },
    {
        "relative": "CricaVPR_10752/CricaVPR.pth",
        "min_bytes": 500_000_000,
        "url": (
            "https://github.com/Lu-Feng/CricaVPR/releases/download/v1.0/" "CricaVPR.pth"
        ),
    },
    {
        "relative": "SuperVLAD_3072/dinov2_vitb14_pretrain.pth",
        "min_bytes": 300_000_000,
        "copy_from": "CricaVPR_10752/dinov2_vitb14_pretrain.pth",
    },
    {
        "relative": "SuperVLAD_3072/SuperVLAD.pth",
        "min_bytes": 600_000_000,
        "gdrive_id": "1yomnWGTJko6nf3F2Ju6RWsLhP2EG82tL",
    },
]


def _ready(path, minimum):
    return os.path.isfile(path) and os.path.getsize(path) >= minimum


def _copy(source, target):
    print(f"[checkpoint] copy {source} -> {target}", flush=True)
    shutil.copy2(source, target + ".part")
    os.replace(target + ".part", target)


def _download(spec, target):
    part = target + ".part"
    if os.path.exists(part):
        os.remove(part)
    if spec.get("gdrive_id"):
        print(f"[checkpoint] download Google Drive id={spec['gdrive_id']}", flush=True)
        budget = float(os.getenv("CHECKPOINT_TOTAL_TIMEOUT_S", "1800"))
        try:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "gdown",
                    "--id",
                    spec["gdrive_id"],
                    "-O",
                    part,
                    "--no-cookies",
                ],
                check=True,
                timeout=budget,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"gdown exceeded {budget:g}s; treating as unreachable"
            ) from exc
        if not os.path.exists(part) or os.path.getsize(part) == 0:
            raise RuntimeError("gdown produced no file")
    else:
        print(f"[checkpoint] download {spec['url']}", flush=True)
        timeout_s = float(os.getenv("CHECKPOINT_TIMEOUT_S", "60"))
        with contextlib.closing(
            urllib.request.urlopen(spec["url"], timeout=timeout_s)
        ) as resp, open(part, "wb") as fh:
            shutil.copyfileobj(resp, fh)
    os.replace(part, target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        default=os.getenv("STAR_VPR_PRETRAINED_SOURCE"),
        help="optional offline assets root with the same model-directory layout",
    )
    parser.add_argument(
        "--record-digests",
        action="store_true",
        help="(re)write configs/checkpoint_digests.json from this host's weights. "
        "Run it on the machine whose numbers are the reference; every other "
        "machine then compares against the committed file.",
    )
    args = parser.parse_args()

    assets = os.path.join(ROOT, "assets")
    failures = []
    for spec in FILES:
        target = os.path.join(assets, spec["relative"])
        if _ready(target, spec["min_bytes"]):
            print(f"[skip checkpoint] {spec['relative']}")
            continue
        os.makedirs(os.path.dirname(target), exist_ok=True)
        try:
            source = None
            if args.source_root:
                candidate = os.path.join(args.source_root, spec["relative"])
                if _ready(candidate, spec["min_bytes"]):
                    source = candidate
            if source is None and spec.get("vendored"):
                candidate = os.path.join(ROOT, spec["vendored"])
                if _ready(candidate, spec["min_bytes"]):
                    source = candidate
            if source is None and spec.get("copy_from"):
                candidate = os.path.join(assets, spec["copy_from"])
                if _ready(candidate, spec["min_bytes"]):
                    source = candidate
            if source:
                _copy(source, target)
            else:
                _download(spec, target)
            if not _ready(target, spec["min_bytes"]):
                raise RuntimeError(
                    f"file is smaller than expected ({os.path.getsize(target)} bytes)"
                )
        except Exception as exc:
            failures.append(f"{spec['relative']}: {type(exc).__name__}: {exc}")
            print(f"[checkpoint error] {failures[-1]}", file=sys.stderr)

    if failures:
        raise SystemExit(
            "public checkpoint preparation incomplete:\n  - " + "\n  - ".join(failures)
        )
    print("[checkpoint ok] all local-weight Table-II backbones are ready")

    current = resolve_digests()
    if args.record_digests:
        os.makedirs(os.path.dirname(DIGEST_FILE), exist_ok=True)
        with open(DIGEST_FILE, "w", encoding="utf-8") as fh:
            json.dump(current, fh, indent=2, sort_keys=True)
        print(
            f"[checkpoint digests] recorded {len(current)} weights -> "
            f"{os.path.relpath(DIGEST_FILE, ROOT)}"
        )
        return

    if not os.path.isfile(DIGEST_FILE):
        print(
            "[checkpoint digests] no reference file; run once with "
            "--record-digests on the reference machine"
        )
        return
    with open(DIGEST_FILE, "r", encoding="utf-8") as fh:
        recorded = json.load(fh)
    mismatched, missing = compare_digests(current, recorded)
    if mismatched:
        raise SystemExit(
            "checkpoint digest mismatch; refusing to build paper caches:\n  - "
            + "\n  - ".join(mismatched)
        )
    if missing:
        print(f"[checkpoint digests] not present on this host: {', '.join(missing)}")
    if not mismatched:
        print(
            f"[checkpoint digests] {len(recorded) - len(missing)} weights match "
            "the reference"
        )


if __name__ == "__main__":
    main()
