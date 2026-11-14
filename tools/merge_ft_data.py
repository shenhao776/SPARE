#!/usr/bin/env python


import argparse
import os

SCENES = [
    "RobotCar",
    "BusinessCampus",
    "Neighborhood",
    "OldTown",
    "NCLT",
    "Boreas",
    "CustomCampus",
]


def _parse(name):
    parts = name.split("@")
    return float(parts[1]), float(parts[2]), parts[-1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, help="root of the per-scene exports")
    ap.add_argument("--name", default="UnionMaps")
    ap.add_argument("--scenes", default=",".join(SCENES))
    ap.add_argument("--offset", type=float, default=1.0e7, help="metres per scene")
    a = ap.parse_args()
    scenes = [s for s in a.scenes.split(",") if s]
    out = os.path.join(a.src, a.name, "images")
    if os.path.isdir(out):
        raise SystemExit(f"{out} exists; remove it first (it holds only symlinks)")
    for split in ("train", "val"):
        for kind in ("database", "queries"):
            dst = os.path.join(out, split, kind)
            os.makedirs(dst)
            total = 0
            for index, scene in enumerate(scenes):
                src = os.path.join(a.src, scene, "images", split, kind)
                names = sorted(os.listdir(src))
                xs = [_parse(n)[0] for n in names]
                extent = max(xs) - min(xs)
                if extent >= a.offset:
                    raise SystemExit(f"{scene}: x extent {extent:.0f} m >= offset")
                base = min(xs)
                for n in names:
                    x, y, stem = _parse(n)
                    new = (
                        f"@{x - base + index * a.offset:.3f}@{y:.3f}"
                        + "@" * 11
                        + f"@{scene}__{stem}"
                    )
                    os.symlink(
                        os.path.realpath(os.path.join(src, n)), os.path.join(dst, new)
                    )
                total += len(names)
            print(f"  {split}/{kind}: {total} links from {len(scenes)} scenes")
        os.symlink("database", os.path.join(out, split, "gallery"))
    os.symlink("val", os.path.join(out, "test"))
    print(f"merged -> {out}")


if __name__ == "__main__":
    main()
