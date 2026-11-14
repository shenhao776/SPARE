#!/usr/bin/env python3


import csv
import os
import re
import sys
import tempfile
import time

os.environ.setdefault("DATASET", "robotcar")
ROOT = os.path.dirname(os.path.abspath(__file__))
if not os.path.isdir(os.path.join(ROOT, "configs")):
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import quick_reproduce as qr
import run_paper_tables as rpt

rpt.FORCE_CPU = True
SCENE = "RobotCar"
FAILURES = 0


def _mark(ok):
    global FAILURES
    FAILURES += not ok
    return "ok" if ok else "FAIL"


NEAR = 0


def _mark3(exact, near):

    global NEAR
    if exact:
        return _mark(True)
    if near:
        NEAR += 1
        return "≈"
    return _mark(False)


def _near(a, b):

    if a is None or b is None:
        return False
    from decimal import Decimal

    return abs(Decimal(rpt._fmt1(float(a))) - Decimal(rpt._fmt1(float(b)))) <= Decimal(
        "0.1"
    )


def _close(a, b):

    if a is None or b is None:
        return False
    return rpt._fmt1(float(a)) == rpt._fmt1(float(b))


def _parse_cell(text):

    out = []
    for part in text.split("/"):
        m = re.fullmatch(r"([0-9.]+)(?:±([0-9.]+))?", part.strip())
        out.append((float(m.group(1)), float(m.group(2)) if m.group(2) else None))
    return out


def _credential(name):
    with open(os.path.join(ROOT, "results", "tables", name), newline="") as fh:
        return list(csv.reader(fh))


def table3():
    print(
        "\n## Cross-retriever transfer, RobotCar slice of the Table 3 trajectory mean (5/15 m shown)"
    )
    rows = _credential("backbones.csv")
    header, data = rows[0], [r for r in rows[1:] if r and r[0] == SCENE]
    backbones = header[2:]
    expect = {r[1]: dict(zip(backbones, r[2:])) for r in data}
    per = {}
    for backbone in backbones:
        per[backbone] = {
            th: rpt._own_dataset_metrics(
                SCENE, backbone, rpt.FULL_VARIANT, LOG, seed_average=True, threshold=th
            )
            for th in (5.0, 10.0, 15.0)
        }
    rpt._progress_clear()
    print(f"| {'Backbone':<12} | {'Raw':>10} | {'+SPARE':>10} | check |")
    print("|" + "-" * 14 + "|" + ("-" * 12 + "|") * 2 + "-------|")
    for backbone in backbones:
        cells = {}
        exact = True
        near = True
        for kind in ("Raw", "+SPARE"):
            got = []
            for th in (5.0, 10.0, 15.0):
                m = per[backbone][th]
                v = (
                    m.get(f"Raw VPR Base: {backbone}")
                    if kind == "Raw"
                    else rpt._metric_by_prefix(m, "Ours (")
                )
                got.append(v)
            exp = [mean for mean, _ in _parse_cell(expect[kind][backbone])]
            exact &= all(_close(g, e) for g, e in zip(got, exp))
            near &= all(
                (_close if kind == "Raw" else _near)(g, e) for g, e in zip(got, exp)
            )
            cells[kind] = (
                f"{rpt._fmt1(got[0])}/{rpt._fmt1(got[2])}"
                if None not in got
                else "MISS"
            )
        print(
            f"| {backbone:<12} | {cells['Raw']:>10} | {cells['+SPARE']:>10} "
            f"| {_mark3(exact, near):^5} |"
        )


def table5():
    print("\n## 25 m success radius, RobotCar column")
    rows = _credential("threshold25.csv")
    col = rows[0].index(SCENE)
    expect = {r[0]: r[col] for r in rows[1:]}
    m25 = rpt._own_dataset_metrics(
        SCENE, "MegaLoc", rpt.FULL_VARIANT, LOG, seed_average=True, threshold=25.0
    )
    pool = rpt.MODEL_IDS[rpt.MAIN_BACKBONE]
    got = {
        "Raw MegaLoc": m25.get("Raw VPR Base: MegaLoc"),
        "Single-frame dense GV (LightGlue)": m25.get(
            "Single-Frame Dense GV (LightGlue)"
        ),
        "FoL": rpt._errors_recall(
            rpt._baseline_json("fol_rerank", SCENE, pool), "reranked_errors_m", 25.0
        ),
        "Pair-VPR (stage 2)": rpt._errors_recall(
            rpt._baseline_json("pairvpr_s2", SCENE, pool), "reranked_errors_m", 25.0
        ),
        "SPARE (ours)": rpt._metric_by_prefix(m25, "Ours ("),
        "Pool upper bound (Top-10)": m25.get("Recall@K (pool ceiling)"),
    }
    seeds = m25.get("__star_seeds") or []
    star_pstd = None
    if len(seeds) > 1:
        mu = sum(seeds) / len(seeds)
        star_pstd = (sum((x - mu) ** 2 for x in seeds) / len(seeds)) ** 0.5
    rpt._progress_clear()
    print(f"| {'Method':<34} | {'25 m':>10} | check |")
    print("|" + "-" * 36 + "|" + "-" * 12 + "|-------|")
    for label, exp_text in expect.items():
        mean, _, std = exp_text.partition("+-")
        g = got.get(label)
        ok = _close(g, mean)
        cell = rpt._fmt1(g) if g is not None else "MISS"
        if std:
            ok &= _close(star_pstd, std)
            cell += f"+-{rpt._fmt1(star_pstd)}" if star_pstd is not None else "+-?"
        print(f"| {label:<34} | {cell:>10} | {_mark(ok):^5} |")


def table6():
    print("\n## Query-FOV sweep on RobotCar test crops (MegaLoc pool, 5 m)")
    rows = _credential("crop_sweep.csv")
    header, data = rows[0], rows[1:]
    scenes = [
        "RobotCar",
        "RobotCarCa4050",
        "RobotCarCa5060",
        "RobotCarCa7080",
        "RobotCarCfull",
    ]
    pool = rpt.MODEL_IDS["MegaLoc"]
    slug_key = {
        "Raw MegaLoc": ("fol_rerank", "raw_errors_m"),
        "FoL": ("fol_rerank", "reranked_errors_m"),
        "PairVPR-s2": ("pairvpr_s2", "reranked_errors_m"),
    }
    heads = {
        "SPARE (universal head)": "UnionMaps",
        "SPARE (universal mixed-crop head)": "UnionMapsMix",
    }
    star = {}
    for label, weight_scene in heads.items():
        for scene in scenes:
            m = rpt._run_own_partition(
                scene,
                "MegaLoc",
                rpt.FULL_VARIANT,
                LOG,
                seed_average=True,
                threshold=5.0,
                weight_scene=weight_scene,
            )
            seeds = m.get("__star_seeds") or []
            mu = sum(seeds) / len(seeds) if seeds else None
            std = None
            if len(seeds) > 1:
                std = (sum((x - mu) ** 2 for x in seeds) / (len(seeds) - 1)) ** 0.5
            star[(label, scene)] = (mu, std)
    rpt._progress_clear()
    print(
        f"| {'Method':<34} | "
        + " | ".join(f"{h:>14}" for h in header[1:])
        + " | check |"
    )
    for row in data:
        label = row[0]
        ok = True
        cells = []
        for scene, exp_text in zip(scenes, row[1:]):
            exp_mean, exp_std = _parse_cell(exp_text)[0]
            if label in heads:
                g, gs = star[(label, scene)]
                ok &= _close(g, exp_mean) and (exp_std is None or _close(gs, exp_std))
                cells.append(
                    f"{rpt._fmt1(g)}±{rpt._fmt1(gs)}" if g is not None else "MISS"
                )
            else:
                slug, key = slug_key[label]
                g = rpt._errors_recall(rpt._baseline_json(slug, scene, pool), key, 5.0)
                ok &= _close(g, exp_mean)
                cells.append(rpt._fmt1(g))
        print(
            f"| {label:<34} | "
            + " | ".join(f"{c:>14}" for c in cells)
            + f" | {_mark(ok):^5} |"
        )


def main():
    global LOG
    start = time.time()
    LOG = tempfile.mkstemp(prefix="robotcar_reproduce_", suffix=".log")[1]
    print(
        "SPARE RobotCar reproduction -- RobotCar slice of the released tables, CPU only"
    )
    print(f"(same evaluation code as the full replay; detailed log: {LOG})")
    print("\n## Main comparison (paper Table 1), RobotCar column")
    global FAILURES
    FAILURES += 1 if qr.main() else 0
    table3()
    table5()
    table6()
    print(f"\nDone in {time.time() - start:.0f}s on CPU.")
    if NEAR:
        print(
            f"≈ : {NEAR} row(s) match the published value up to one near-tied "
            "candidate per seed (single-thread CPU vs the canonical CUDA replay; "
            "see README)."
        )
    if FAILURES:
        print(f"FAIL -- {FAILURES} row(s) differ from the published tables.")
        return 1
    print("PASS -- every RobotCar number matches the published tables.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
