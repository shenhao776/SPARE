#!/usr/bin/env python


import argparse
import bisect
import json
import os
import sys
import tempfile
from decimal import Decimal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IMG_EXT = (".png", ".jpg", ".jpeg", ".tiff")
POSE_FILES = ("pose.txt", "GNSSPoses.txt")
MANIFEST = ".clean_manifest.json"


def ts_to_ns(ts_str):
    ts_str = ts_str.strip().split("@")[0]
    if "." in ts_str:
        return int(Decimal(ts_str) * Decimal("1e9"))
    val, n = int(ts_str), len(ts_str)
    scale = {19: 1, 16: 1_000, 13: 1_000_000, 10: 1_000_000_000}.get(n)
    if scale is None:
        raise ValueError(f"unrecognised timestamp '{ts_str}' ({n} digits)")
    return val * scale


def find_pose_file(img_dir, stop_at):

    cur = img_dir
    while True:
        for name in POSE_FILES:
            p = os.path.join(cur, name)
            if os.path.exists(p):
                return p
        if os.path.samefile(cur, stop_at):
            return None
        parent = os.path.dirname(cur)
        if parent == cur:
            return None
        cur = parent


def load_pose_ns(img_dir, stop_at):

    p = find_pose_file(img_dir, stop_at)
    if p is None:
        return None
    out = []
    with open(p, encoding="utf-8") as handle:
        for ln in handle:
            ln = ln.strip()
            if not ln or ln.startswith("#") or ln.startswith("timestamp"):
                continue
            try:
                out.append(ts_to_ns(ln.replace(",", " ").split()[0]))
            except ValueError:
                continue
    return sorted(out)


def write_json_atomic(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=os.path.basename(path) + ".", suffix=".tmp", dir=os.path.dirname(path)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=1, ensure_ascii=False)
            handle.write("\n")
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def remove_stale_outputs(out, plan):

    if not os.path.isdir(out):
        return
    expected = {os.path.relpath(dst, out).replace(os.sep, "/") for _src, dst in plan}
    expected.add(MANIFEST)
    removed = 0
    for dirpath, dirnames, files in os.walk(out, topdown=False):
        for name in files:
            path = os.path.join(dirpath, name)
            rel = os.path.relpath(path, out).replace(os.sep, "/")
            if rel not in expected:
                os.unlink(path)
                removed += 1
        for name in dirnames:
            path = os.path.join(dirpath, name)
            rel = os.path.relpath(path, out).replace(os.sep, "/")
            if os.path.islink(path) and rel not in expected:
                os.unlink(path)
                removed += 1
            elif os.path.isdir(path) and not os.listdir(path):
                os.rmdir(path)
    if removed:
        print(f"[clean] removed {removed} stale derived entries from {out}")


def _has_clean_images(out):
    if not os.path.isdir(out):
        return False
    return any(
        name.lower().endswith(IMG_EXT)
        for _dirpath, _dirs, files in os.walk(out)
        for name in files
    )


def clean(src, out, tol_ms, dry_run):
    src, out = os.path.abspath(src), os.path.abspath(out)
    assert os.path.isdir(src), f"missing source: {src}"
    assert (
        not out.startswith(src + os.sep) and out != src
    ), "output view must live outside the source tree"

    mpath = os.path.join(out, MANIFEST)
    if _has_clean_images(out):
        print(f"[clean] existing derived view; presence cache hit -> {out}")
        return 0

    tol_ns = tol_ms * 1_000_000
    kept, rejected = 0, []
    plan = []
    for dirpath, _, files in os.walk(src, followlinks=True):
        rel = os.path.relpath(dirpath, src)
        imgs = [f for f in files if f.lower().endswith(IMG_EXT)]
        metas = [f for f in files if not f.lower().endswith(IMG_EXT)]
        if not imgs and not metas:
            continue
        pose_ns = load_pose_ns(dirpath, src) if imgs else None
        for f in metas:
            plan.append((os.path.join(dirpath, f), os.path.join(out, rel, f)))
        for f in imgs:
            why = None
            if not pose_ns:
                why = "no-pose-file"
            else:
                try:
                    ns = ts_to_ns(os.path.splitext(f)[0])
                    i = bisect.bisect_left(pose_ns, ns)
                    near = min(
                        (
                            abs(pose_ns[j] - ns)
                            for j in (i - 1, i)
                            if 0 <= j < len(pose_ns)
                        ),
                        default=None,
                    )
                    if near is None or near > tol_ns:
                        why = "no-pose-match"
                except ValueError:
                    why = "bad-timestamp"
            if why:
                rejected.append({"file": os.path.join(rel, f), "reason": why})
            else:
                kept += 1
                plan.append((os.path.join(dirpath, f), os.path.join(out, rel, f)))

    print(
        f"[clean] source={src}\n[clean] kept {kept} frames, "
        f"rejected {len(rejected)} ({tol_ms} ms tolerance)"
    )
    for r in rejected[:10]:
        print(f"    - {r['reason']:<14} {r['file']}")
    if len(rejected) > 10:
        print(f"    ... and {len(rejected) - 10} more (full list in manifest)")
    if dry_run:
        print("[clean] dry-run: nothing written")
        return 0

    remove_stale_outputs(out, plan)
    for s, d in plan:
        os.makedirs(os.path.dirname(d), exist_ok=True)
        if os.path.lexists(d):
            os.remove(d)
        os.symlink(os.path.realpath(s), d)
    write_json_atomic(
        mpath,
        {
            "source": src,
            "tolerance_ms": tol_ms,
            "kept": kept,
            "rejected": rejected,
        },
    )
    print(f"[clean] view written -> {out} (source untouched; manifest saved)")
    return 0


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[1] if __doc__ else None
    )
    ap.add_argument("--src", help="raw recording tree (read-only)")
    ap.add_argument("--out", help="output view directory")
    ap.add_argument(
        "--dataset",
        help="scene mode: read src/out from its YAML " "(clean.src -> build.raw_root)",
    )
    ap.add_argument("--tolerance-ms", type=int, default=500)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if a.dataset:
        from core.config import resolve_dataset

        cfg = resolve_dataset(a.dataset)
        cl = cfg.get("clean") or {}
        src = cl.get("src")
        if not src:
            print(
                f"[clean] {cfg['name']}: no `clean:` block in its YAML -- "
                "raw_root is already a final view, nothing to do"
            )
            return 0
        out = cfg["build"]["raw_root"]
        out = out if os.path.isabs(out) else os.path.join(ROOT, out)
        return clean(src, out, int(cl.get("tolerance_ms", a.tolerance_ms)), a.dry_run)
    assert a.src and a.out, "need --src and --out (or --dataset)"
    return clean(a.src, a.out, a.tolerance_ms, a.dry_run)


if __name__ == "__main__":
    sys.exit(main())
