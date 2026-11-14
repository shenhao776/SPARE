#!/usr/bin/env python3


import csv
import os
import sys
import tempfile
import time

os.environ.setdefault("DATASET", "robotcar")
ROOT = os.path.dirname(os.path.abspath(__file__))
if not os.path.isdir(os.path.join(ROOT, "configs")):
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import run_paper_tables as rpt

rpt.FORCE_CPU = True

SCENE, BACKBONE = "RobotCar", "MegaLoc"
ROWS = list(rpt.PAPER_CONFIG["main_rows"])


def _csv_key(spec):

    for entry in rpt.PAPER_CONFIG["paper_csv"]["main"]:
        src = entry.get("from")
        if (
            spec["kind"] == "star"
            and src == "backbone"
            and entry.get("kind") == "+SPARE"
        ):
            return entry["method"]
        if (
            spec["label"] == "Raw MegaLoc"
            and src == "backbone"
            and entry.get("kind") == "Raw"
        ):
            return entry["method"]
        if src == "main_row" and entry.get("label") == spec["label"]:
            return entry["method"]
        if src == "learned" and entry.get("method_key") == spec.get("method"):
            return entry["method"]
    return None


def _published():

    path = os.path.join(ROOT, "results", "tables", "main.csv")
    with open(path, encoding="utf-8") as fh:
        by_method = {r["method"]: r for r in csv.DictReader(fh)}
    mean = {}
    for spec in ROWS:
        key = _csv_key(spec)
        if key is None or key not in by_method:
            continue
        cell = by_method[key][SCENE]
        mean[spec["label"]] = tuple(float(p.split("±")[0]) for p in cell.split("/"))
    return mean


EXPECTED = _published()
STD_BOUND = 0.9


def _same(got, exp):
    return got is not None and rpt._fmt1(float(got)) == rpt._fmt1(float(exp))


def main():
    start = time.time()
    log_path = tempfile.mkstemp(prefix="quick_reproduce_", suffix=".log")[1]
    print(f"SPARE quick reproduction -- paper main table, {SCENE} column, CPU only")
    print(f"(same evaluation code as the full replay; detailed log: {log_path})\n")

    radii = rpt._sweep_order(SCENE)
    per_th = {
        th: rpt._safe_metrics(
            SCENE, BACKBONE, rpt.FULL_VARIANT, log_path, seed_average=True, threshold=th
        )
        for th in radii
    }
    pool = rpt.MODEL_IDS[rpt.MAIN_BACKBONE]

    failures = 0
    rpt._progress_clear()
    print(f"| {'Method':<34} | {'5 m':>6} | {'10 m':>6} | {'15 m':>6} | check |")
    print("|" + "-" * 36 + "|" + ("-" * 8 + "|") * 3 + "-------|")
    for spec in ROWS:
        label, kind = spec["label"], spec["kind"]
        if label not in EXPECTED:
            continue
        if kind == "metric":
            triple = [rpt._metric_by_prefix(per_th[th], spec["prefix"]) for th in radii]
        elif kind == "star":
            triple = [rpt._metric_by_prefix(per_th[th], "Ours (") for th in radii]
        elif kind == "classic":
            triple = (rpt._baseline_values(spec["method"], [SCENE]) or {}).get(SCENE)
        else:
            vals = rpt._learned_reranker_values(
                spec["method"], [SCENE], pool=spec.get("pool") or pool
            )
            triple = (vals or {}).get(SCENE)
        triple = list(triple or [None] * 3)
        ok, cells = True, []
        for i, got in enumerate(triple):
            if got is None:
                cells.append("MISS")
                ok = False
                continue
            if not _same(got, EXPECTED[label][i]):
                ok = False
            cells.append(rpt._fmt1(float(got)))
        if kind == "star":
            for i, th in enumerate(radii):
                std = (per_th[th] or {}).get("__star_std")
                if std is None or float(rpt._fmt1(float(std))) > STD_BOUND:
                    ok = False
        failures += not ok
        mark = "ok" if ok else "FAIL"
        print(
            f"| {label:<34} | {cells[0]:>6} | {cells[1]:>6} | {cells[2]:>6} | {mark:^5} |"
        )

    print(f"\nDone in {time.time() - start:.0f}s on CPU.")
    if failures:
        print(f"FAIL -- {failures} row(s) differ from the published numbers.")
        return 1
    print(f"PASS -- every number matches the paper (main table, {SCENE} column).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
