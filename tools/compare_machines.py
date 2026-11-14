#!/usr/bin/env python


import argparse
import json
import sys


def load(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    args = ap.parse_args()
    a, b = load(args.a), load(args.b)

    pa, pb = a.get("provenance", {}), b.get("provenance", {})
    name_a = pa.get("host", args.a)
    name_b = pb.get("host", args.b)
    print(
        f"A = {name_a}   commit {pa.get('commit')}{' +dirty' if pa.get('dirty') else ''}"
    )
    print(
        f"B = {name_b}   commit {pb.get('commit')}{' +dirty' if pb.get('dirty') else ''}"
    )
    print("=" * 76)

    fatal = []
    ca, cb = str(pa.get("commit") or ""), str(pb.get("commit") or "")
    same = bool(ca) and bool(cb) and (ca.startswith(cb) or cb.startswith(ca))
    if not same:
        fatal.append(
            f"different code: {pa.get('commit')} vs {pb.get('commit')}. "
            "Sync the checkouts before comparing anything else."
        )
    for label, prov in ((name_a, pa), (name_b, pb)):
        if prov.get("dirty"):
            fatal.append(
                f"{label} has uncommitted changes; its results are not "
                "attributable to any commit."
            )
    if fatal:
        for line in fatal:
            print(f"❌ {line}")
        return 1
    print("✅ same commit, both clean")

    sa, sb = pa.get("stack") or {}, pb.get("stack") or {}
    libs = [k for k in sorted(set(sa) | set(sb)) if k != "gpu"]
    mismatch = [
        f"{k}: {sa.get(k)} vs {sb.get(k)}" for k in libs if sa.get(k) != sb.get(k)
    ]
    if mismatch:
        print(
            "⚠️  software stack differs -- expect layer-③ differences beyond GPU noise:"
        )
        for line in mismatch:
            print(f"   {line}")
    elif libs:
        print(f"✅ same software stack ({len(libs)} libraries)")
    if sa.get("gpu") != sb.get("gpu"):
        print(
            f"·  different GPU ({sa.get('gpu')} vs {sb.get('gpu')}) -- float kernel "
            "selection is architecture-dependent, so bit-identical layer ③ is "
            "not expected"
        )

    da, db = a.get("layer2_digests", {}), b.get("layer2_digests", {})
    drift = []
    for scene in sorted(set(da) | set(db)):
        for split in sorted(set(da.get(scene, {})) | set(db.get(scene, {}))):
            x, y = da.get(scene, {}).get(split), db.get(scene, {}).get(split)
            if x != y:
                drift.append(f"{scene}/{split}: {x} vs {y}")
    if drift:
        print(
            f"⚠️  layer ② differs in {len(drift)} split(s) -- the two machines' "
            "pixels are not byte-identical, so read every layer-③ difference "
            "below as 'encoder + GPU', not 'GPU':"
        )
        for line in drift[:10]:
            print(f"   {line}")
        print(
            "   Most likely a different OpenCV/libjpeg build (see "
            "env/clean_env.yml) or different CROP_* settings."
        )
    else:
        print(f"✅ layer ② identical across {len(da)} scenes")

    sa, sb = a.get("state_caches", {}), b.get("state_caches", {})
    only_a = sorted(set(sa) - set(sb))
    only_b = sorted(set(sb) - set(sa))
    differing = []
    for key in sorted(set(sa) & set(sb)):
        if sa[key] != sb[key]:
            differing.append(
                f"{key}: A rows={sa[key]['rows']} q={sa[key]['queries']} "
                f"deg={sa[key]['degenerate']} | B rows={sb[key]['rows']} "
                f"q={sb[key]['queries']} deg={sb[key]['degenerate']}"
            )
    print(
        f"layer ③ state caches: {len(set(sa) & set(sb))} shared, "
        f"{len(only_a)} only on A, {len(only_b)} only on B, "
        f"{len(differing)} with different shape"
    )
    for line in differing[:10]:
        print(f"   ⚠️  {line}")
    for key in only_a[:5]:
        print(f"   · only on {name_a}: {key}")
    for key in only_b[:5]:
        print(f"   · only on {name_b}: {key}")

    print(
        f"problems: {name_a} {len(a.get('problems', []))}, "
        f"{name_b} {len(b.get('problems', []))}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
