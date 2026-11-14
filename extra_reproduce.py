#!/usr/bin/env python3


import argparse
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
import time

os.environ.setdefault("DATASET", "robotcar")
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import run_paper_tables as rpt

LOG = None
TOL = 1.0
COUNTS = {"ok": 0, "≈": 0, "FAIL": 0}
POOL = "MegaLoc_8448"
CROP_SUFFIXES = ["", "Ca4050", "Ca5060", "Ca7080", "Cfull"]


def _num(text):
    m = re.match(r"\s*(-?[0-9.]+)", str(text))
    return float(m.group(1)) if m else None


def _judge(got, expected, tol=None):

    tol = TOL if tol is None else tol
    if got == expected:
        mark = "ok"
    else:
        g, e = _num(got), _num(expected)
        mark = "≈" if g is not None and e is not None and abs(g - e) <= tol else "FAIL"
    COUNTS[mark] += 1
    return mark


def _credential(name):
    with open(
        os.path.join(ROOT, "results", "tables", name), newline="", encoding="utf-8"
    ) as fh:
        return list(csv.reader(fh))


def _seed_cell(scene, threshold, weight_scene, variant=None):
    m = rpt._run_own_partition(
        scene,
        "MegaLoc",
        variant or rpt.FULL_VARIANT,
        LOG,
        seed_average=True,
        threshold=threshold,
        weight_scene=weight_scene,
    )
    seeds = m.get("__star_seeds") or []
    if not seeds:
        return "MISS", m
    mean = sum(seeds) / len(seeds)
    std = (
        (sum((x - mean) ** 2 for x in seeds) / (len(seeds) - 1)) ** 0.5
        if len(seeds) > 1
        else 0.0
    )
    return f"{rpt._fmt1(mean)}±{rpt._fmt1(std)}", m


def _print_rows(title, header, rows):
    rpt._progress_clear()
    print(f"\n## {title}")
    widths = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
    for r in [header] + rows:
        print("| " + " | ".join(str(c).ljust(w) for c, w in zip(r, widths)) + " |")


def part_fov():

    finetuned = (
        ("GeoWarp (fine-tuned)", "geowarp_target"),
        ("SelaVPR (fine-tuned)", "selavpr_target"),
        ("PairVPR-s2 (fine-tuned)", "pairvpr_s2_target"),
    )
    target = {(r[0], r[1]): r[2:] for r in _credential("fov_finetuned.csv")[1:]}
    for bench, stem in (("Pitts30k", "pitts30k"), ("MSLS", "msls")):
        cred = _credential(f"crop_sweep_{stem}.csv")
        header, expected = cred[0], {r[0]: r[1:] for r in cred[1:]}
        for label, _ in finetuned:
            expected[label] = target[(label, bench)]
        scenes = [bench + s for s in CROP_SUFFIXES]
        got = {}
        for scene in scenes:
            cell, _ = _seed_cell(scene, 25.0, "UnionMaps")
            got.setdefault("SPARE (universal head)", []).append(cell)
            cell, _ = _seed_cell(scene, 25.0, "UnionMapsMix")
            got.setdefault("SPARE (universal mixed-crop head)", []).append(cell)
            for label, slug, key in (
                ("Raw MegaLoc", "fol_rerank", "raw_errors_m"),
                ("FoL", "fol_rerank", "reranked_errors_m"),
                ("PairVPR-s2", "pairvpr_s2", "reranked_errors_m"),
            ):
                v = rpt._errors_recall(rpt._baseline_json(slug, scene, POOL), key, 25.0)
                got.setdefault(label, []).append(rpt._fmt1(v))
            for label, slug in finetuned:
                with open(
                    rpt._baseline_json(slug, scene, POOL), encoding="utf-8"
                ) as fh:
                    v = json.load(fh)["reranked_recall"]["25"]
                got.setdefault(label, []).append(rpt._fmt1(v))
        rows = []
        order = [
            "Raw MegaLoc",
            "FoL",
            "PairVPR-s2",
            *[label for label, _ in finetuned],
            "SPARE (universal head)",
            "SPARE (universal mixed-crop head)",
        ]
        for label in order:
            cells = [f"{g} {_judge(g, e)}" for g, e in zip(got[label], expected[label])]
            rows.append([label] + cells)
        _print_rows(
            f"Table 5 -- query field of view, {bench}, 25 m (got + check)",
            ["method"] + header[1:],
            rows,
        )


def part_channels():

    arms = [
        ("universal", "UnionMaps", {}),
        ("universal w/o zoom", "UnionMaps", {"ABLATION_USE_ZOOM": "False"}),
        ("mixed-crop", "UnionMapsMix", {}),
    ]
    for bench, suffix, th in (
        ("RobotCar", "", 5.0),
        ("Pitts30k", "_pitts30k", 25.0),
        ("MSLS", "_msls", 25.0),
    ):
        cred = _credential(f"fov_channel_ablation{suffix}.csv")
        header, expected = cred[0], {r[0]: r[1:] for r in cred[1:]}
        scenes = [bench + s for s in CROP_SUFFIXES]
        rows = []
        for label, ws, over in arms:
            variant = dict(rpt.FULL_VARIANT, **over)
            cells = []
            for scene, exp in zip(scenes, expected[label]):
                g, _ = _seed_cell(scene, th, ws, variant)
                cells.append(f"{g} {_judge(g, exp)}")
            rows.append([label] + cells)
        _print_rows(
            f"Field-of-view channel ablation, {bench}, {th:g} m (got + check)",
            header,
            rows,
        )


TIMING = os.path.join("assets", "timing_20261109")


def _bench_blocks(path):

    runs, cur = [], {}
    labels = (
        ("match", "LightGlue"),
        ("geom", "RANSAC/几何"),
        ("dino", "DINOv2"),
        ("vprp", "MegaLoc patch"),
    )
    for line in open(path, encoding="utf-8", errors="replace"):
        for key, lab in labels:
            m = re.match(r"\s*" + re.escape(lab) + r"\s+(\d+)\s*ms", line)
            if m:
                cur[key] = float(m.group(1))
        m = re.match(r"\s*总计\s+(\d+)\s*ms", line)
        if m:
            cur["total"] = float(m.group(1))
            runs.append(cur)
            cur = {}
    return runs


def _fmt_ms(ms):
    return f"{ms / 1000:.1f} s" if ms >= 1000 else f"{ms:.0f} ms"


def part_latency():

    expected = {r[0]: r[1] for r in _credential("latency.csv")[1:]}
    runs = [
        r
        for r in _bench_blocks(os.path.join(TIMING, "timing.log"))
        if r.get("vprp", 0) > 0
    ]
    spare = runs[-1]
    got = {
        "SPARE (ours)": _fmt_ms(spare["total"]),
        "SPARE stages (LightGlue / RANSAC / DINOv2 / MegaLoc patches)": " / ".join(
            f"{spare[k]:.0f}" for k in ("match", "geom", "dino", "vprp")
        )
        + " ms",
    }
    for label, stem in (
        ("VOP", "vop_rerank"),
        ("R2Former", "r2former_rerank"),
        ("FoL", "fol_rerank"),
        ("SelaVPR", "selavpr_rerank"),
        ("Pair-VPR (stage 2)", "pairvpr_s2"),
        ("GeoWarp", "geowarp_rerank"),
    ):
        with open(
            os.path.join(TIMING, f"{stem}_RobotCar__pool-{POOL}.json"), encoding="utf-8"
        ) as fh:
            d = json.load(fh)
        got[label] = _fmt_ms(1000.0 * float(d["elapsed_s"]) / float(d["n_queries"]))
    with open(os.path.join(TIMING, "roma_bench.json"), encoding="utf-8") as fh:
        got["RoMa GV"] = _fmt_ms(float(json.load(fh)["ms_per_query"]))
    rows = []
    for label, exp in expected.items():
        g = got[label]
        tol = 0.05 * (_num(exp) or 0)
        rows.append([label, g, exp, _judge(g, exp, tol)])
    _print_rows(
        "Table 7 -- re-ranking time per query (recorded run)",
        ["method", "got", "published", "check"],
        rows,
    )


def part_standard():

    cred = _credential("public_benchmark.csv")
    exp = {r[0]: dict(zip(cred[0], r)) for r in cred[1:]}["MegaLoc"]
    rows = []
    for bench, key in (("pitts30k", "pitts30k"), ("msls_val", "msls_val")):
        desc = os.path.join(
            "assets", "public_benchmarks", "descriptors", f"MegaLoc_{bench}.npz"
        )
        out = tempfile.mkstemp(prefix=f"std_{bench}_", suffix=".json")[1]
        r = subprocess.run(
            [
                sys.executable,
                "tools/eval_public_benchmark.py",
                "--benchmark",
                bench,
                "--load-desc",
                desc,
                "--out",
                out,
            ],
            capture_output=True,
            text=True,
        )
        if r.returncode:
            raise SystemExit(r.stdout[-2000:] + r.stderr[-2000:])
        with open(out, encoding="utf-8") as fh:
            res = json.load(fh)["MegaLoc"]
        for radius in ("25", "15", "5"):
            if bench == "msls_val" and radius == "25":
                g = rpt._fmt1(res["recall"]["1"])
            else:
                g = rpt._fmt1(res["recall_by_radius"][radius]["1"])
            e = exp[f"{key}_R@1_{radius}m"]
            rows.append([f"{bench} R@1 @ {radius} m", g, e, _judge(g, e)])
    _print_rows(
        "Introduction -- standard evaluation, MegaLoc, uncropped queries",
        ["metric", "got", "published", "check"],
        rows,
    )


def part_fusion():

    out = tempfile.mkdtemp(prefix="rank_fusion_")
    subprocess.run(
        [
            sys.executable,
            os.path.join(ROOT, "tools", "rank_fusion_baseline.py"),
            "--out",
            out,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    with open(os.path.join(out, "rank_fusion.csv"), encoding="utf-8") as fh:
        got = {(r["scene"], r["variant"]): r for r in csv.DictReader(fh)}
    cred = _credential("rank_fusion.csv")
    rows = []
    for values in cred[1:]:
        e = dict(zip(cred[0], values))
        g = got[(e["scene"], e["variant"])]
        gs = "/".join(f"{float(g[k]):.1f}" for k in ("r_a", "r_b", "r_c"))
        es = "/".join(f"{float(e[k]):.1f}" for k in ("r_a", "r_b", "r_c"))
        mark = _judge(gs, es)
        if e["variant"] == "inliers+dino+vprp" and e["scene"] in (
            "Pitts30k",
            "MSLS",
            "CustomCampus",
        ):
            rows.append([e["scene"], gs, es, mark])
    _print_rows(
        "Table 6 -- rank fusion of the three cues (no trained head); "
        f"all {len(cred) - 1} test-set x variant rows compared",
        ["dataset", "got", "published", "check"],
        rows,
    )


PARTS = {
    "fov": part_fov,
    "channels": part_channels,
    "latency": part_latency,
    "standard": part_standard,
    "fusion": part_fusion,
}


def main():
    global LOG, TOL
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--parts", default=",".join(PARTS), help="comma-separated: " + ",".join(PARTS)
    )
    ap.add_argument("--cpu", action="store_true", help="force CPU")
    ap.add_argument(
        "--tol", type=float, default=TOL, help="points allowed for '≈' (default 1.0)"
    )
    a = ap.parse_args()
    TOL = a.tol
    if a.cpu:
        rpt.FORCE_CPU = True
    LOG = tempfile.mkstemp(prefix="extra_reproduce_", suffix=".log")[1]
    start = time.time()
    print(f"SPARE -- paper numbers outside Tables 1-4 and 6 (detailed log: {LOG})")
    for name in a.parts.split(","):
        PARTS[name.strip()]()
    print(
        f"\nDone in {time.time() - start:.0f}s.  ok={COUNTS['ok']}  ≈={COUNTS['≈']}  FAIL={COUNTS['FAIL']}"
    )
    if COUNTS["FAIL"]:
        print(
            "FAIL -- some numbers differ from the published values by more than the tolerance."
        )
        return 1
    print(
        "PASS -- every number matches the published values (≈ = within the tolerance)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
