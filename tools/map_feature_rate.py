#!/usr/bin/env python


import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MEASURES = ["lost_frac", "flow", "flow_p90", "scale", "hull_shift", "view_change"]
IMG_EXT = (".png", ".jpg", ".jpeg", ".tiff")


def _measurement_status(scene, stride, path):
    if not os.path.isfile(path):
        return False, "measurement JSON missing"
    try:
        with open(path, encoding="utf-8") as handle:
            rows = json.load(handle)
    except (OSError, ValueError):
        return False, "measurement JSON unreadable"
    if not isinstance(rows, list) or not rows:
        return False, "measurement JSON empty/invalid"
    return True, "present"


def _ts(filename):

    try:
        return int(filename.split(".")[0].split("@")[0])
    except Exception:
        return 0


def robust(v):

    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan")
    q1, q3 = np.percentile(v, [25, 75])
    return float(np.median(v)), float(q3 - q1)


def measure(scene, model_id, limit, stride):
    import cv2
    import torch
    from core.config import RANSAC_REPROJ_THRESHOLD, FEATURE_MATCH_MIN_RAW
    from core.vpr_core import VPRService, pose_manager, get_all_images_recursive

    db_dir = os.path.join(ROOT, "datasets", scene, "database")
    frames = [os.path.abspath(p) for p in get_all_images_recursive(db_dir)]
    frames = [p for p in frames if p.lower().endswith(IMG_EXT)]
    by_dir = {}
    for p in frames:
        by_dir.setdefault(os.path.dirname(p), []).append(p)
    for d in by_dir:
        by_dir[d].sort(key=lambda p: _ts(os.path.basename(p)))

    pairs = []
    for d, fs in sorted(by_dir.items()):
        for i in range(0, len(fs) - stride, stride):
            pairs.append((fs[i], fs[i + stride]))
    if limit:
        pairs = pairs[:limit]
    print(
        f"[{scene}] {len(frames)} frames in {len(by_dir)} dir(s) -> {len(pairs)} pairs"
    )

    cropper = VPRService().cropper
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    feat_cache = {}

    def feats(path):

        if path in feat_cache:
            return feat_cache[path]
        rel = os.path.relpath(path, os.path.join(ROOT, "datasets", scene))
        asset = os.path.join(ROOT, "assets", "sequence_assets", scene, rel + ".pt")
        out = None
        if os.path.isfile(asset):
            payload = torch.load(asset, map_location="cpu", weights_only=False)
            from sl_agent.cache_fingerprint import validate_sequence_asset

            reason = validate_sequence_asset(payload, rel, scene)
            if reason:
                raise RuntimeError(f"invalid sequence asset {asset}: {reason}")
            fl = payload.get("lightglue_feats") or []
            if fl:
                out = {}
                for k, v in fl[0].items():
                    if hasattr(v, "to"):
                        v = v.to(dev)
                        if v.is_floating_point():
                            v = v.float()
                    out[k] = v
        if len(feat_cache) > 4:
            feat_cache.clear()
        feat_cache[path] = out
        return out

    def rect(f):
        w, h = f["image_size"][0].tolist()
        return np.float32([[0, 0], [w, 0], [w, h], [0, h]]), float(np.hypot(w, h))

    recs, misses = [], 0
    for n, (pa, pb) in enumerate(pairs):
        fa, fb = feats(pa), feats(pb)
        if fa is None or fb is None:
            misses += 1
            continue
        ga = pose_manager.get_pose_info(pa) or {}
        gb = pose_manager.get_pose_info(pb) or {}
        ga, gb = ga.get("global"), gb.get("global")
        dd = (
            abs(pose_manager.calculate_physical_distance(pa, ga, gb) or 0.0)
            if ga and gb
            else np.nan
        )
        dth = (
            abs(pose_manager.calculate_angular_distance(ga, gb) or 0.0)
            if ga and gb
            else np.nan
        )

        kp_a, kp_b, matches = cropper.match_features(fa, fb)
        n_kp = min(int(fa["keypoints"].shape[1]), int(fb["keypoints"].shape[1]))
        rec = {
            "a": os.path.basename(pa),
            "b": os.path.basename(pb),
            "dd": float(dd),
            "dth": float(dth),
            "n_kp": n_kp,
            "raw": len(matches),
            "inliers": 0,
        }
        for k in MEASURES:
            rec[k] = np.nan if k.startswith("flow") or k == "scale" else 1.0
        if len(matches) >= FEATURE_MATCH_MIN_RAW:
            src = np.float32([kp_b[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst = np.float32([kp_a[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            cv2.setRNGSeed(0)
            H, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_REPROJ_THRESHOLD)
            if mask is not None and int(mask.sum()) >= 3:
                keep = [m for k, m in enumerate(matches) if mask[k][0] == 1]
                pa_pts = np.float32([kp_a[m.queryIdx].pt for m in keep])
                pb_pts = np.float32([kp_b[m.trainIdx].pt for m in keep])
                box_a, diag = rect(fa)
                rec["inliers"] = len(keep)
                rec["lost_frac"] = 1.0 - len(keep) / max(n_kp, 1)
                disp = np.linalg.norm(pa_pts - pb_pts, axis=1) / diag
                rec["flow"] = float(np.median(disp))
                rec["flow_p90"] = float(np.percentile(disp, 90))
                hq = cv2.convexHull(pa_pts)
                hs = cv2.convexHull(pb_pts)
                ma, mb = cv2.moments(hq), cv2.moments(hs)
                if ma["m00"] > 1e-9 and mb["m00"] > 1e-9:
                    ca = np.array([ma["m10"] / ma["m00"], ma["m01"] / ma["m00"]])
                    cb = np.array([mb["m10"] / mb["m00"], mb["m01"] / mb["m00"]])
                    rec["hull_shift"] = float(np.linalg.norm(ca - cb)) / diag
                if H is not None and H.shape == (3, 3) and abs(H[2, 2]) > 1e-9:
                    Hn = H / H[2, 2]
                    det = abs(float(np.linalg.det(Hn[:2, :2])))
                    if 1e-6 < det < 1e6:
                        rec["scale"] = abs(float(np.log(np.sqrt(det))))
                    box_b, _ = rect(fb)
                    warped = cv2.perspectiveTransform(
                        box_b.reshape(-1, 1, 2), H
                    ).reshape(-1, 2)
                    if np.isfinite(warped).all() and cv2.isContourConvex(
                        np.float32(warped)
                    ):
                        inter, _ = cv2.intersectConvexConvex(np.float32(warped), box_a)
                        union = (
                            cv2.contourArea(np.float32(warped))
                            + cv2.contourArea(box_a)
                            - inter
                        )
                        if union > 1e-6:
                            rec["view_change"] = float(max(0.0, 1.0 - inter / union))
        recs.append(rec)
        if (n + 1) % 500 == 0:
            print(f"  ...{n + 1}/{len(pairs)}", flush=True)
    if misses:
        print(f"  [warn] {misses} pairs skipped: no cached features")
    return recs


REFERENCE_METRE_STEP = 3.0


def report(scene, recs, interval=REFERENCE_METRE_STEP):
    dd = np.array([r["dd"] for r in recs], float)
    dth = np.array([r["dth"] for r in recs], float)
    ok = np.isfinite(dd)
    print(f"\n===== {scene}: {len(recs)} adjacent pairs =====")
    m, i = robust(dd[ok])
    print(
        f"  frame spacing: median {m:.2f} m (IQR {i:.2f}), "
        f"p90 {np.nanpercentile(dd, 90):.2f}, max {np.nanmax(dd):.1f}"
    )
    m, i = robust(dth[np.isfinite(dth)])
    print(
        f"  frame turning: median {m:.2f} deg (IQR {i:.2f}), "
        f"p90 {np.nanpercentile(dth, 90):.2f}"
    )
    fail = sum(1 for r in recs if r["inliers"] == 0)
    print(f"  unmatched pairs: {fail} ({100.0 * fail / max(len(recs), 1):.1f}%)")

    print(f"\n  -- 同一位移下,变化量的离散度(IQR/中位;越大 = 按米采样越站不住)--")
    edges = [0, 0.5, 1, 2, 3, 5, 8, 1e9]
    hdr = "".join(
        f"{f'{edges[k]:g}-{edges[k+1]:g}m':>12}" for k in range(len(edges) - 1)
    )
    print(f"  {'量':<13}{hdr}")
    for key in MEASURES:
        v = np.array([r.get(key, np.nan) for r in recs], float)
        cells = []
        for k in range(len(edges) - 1):
            sel = ok & (dd >= edges[k]) & (dd < edges[k + 1]) & np.isfinite(v)
            if sel.sum() < 30:
                cells.append(f"{'-':>12}")
                continue
            med, iqr = robust(v[sel])
            cells.append(f"{iqr / max(abs(med), 1e-9):>11.2f} ")
        print(f"  {key:<13}{''.join(cells)}")
    print(
        f"  {'(n)':<13}"
        + "".join(
            f"{int(((dd >= edges[k]) & (dd < edges[k+1]) & ok).sum()):>12}"
            for k in range(len(edges) - 1)
        )
    )

    def spearman(x, y):
        s = np.isfinite(x) & np.isfinite(y)
        if s.sum() < 50:
            return float("nan")
        rx = np.argsort(np.argsort(x[s])).astype(float)
        ry = np.argsort(np.argsort(y[s])).astype(float)
        return float(np.corrcoef(rx, ry)[0, 1])

    print(f"\n  {'量':<13}{'中位':>10}{'IQR':>10}{'ρ(Δd)':>10}{'ρ(Δθ)':>10}")
    for key in MEASURES:
        v = np.array([r.get(key, np.nan) for r in recs], float)
        med, iqr = robust(v)
        print(
            f"  {key:<13}{med:>10.3f}{iqr:>10.3f}"
            f"{spearman(v, dd):>10.2f}{spearman(v, dth):>10.2f}"
        )

    print(
        f"\n  -- 用 tau = 「当前 {interval:g} m 一步的典型变化量」行走,采纳步长分布 --"
    )
    print(
        f"  {'量':<13}{'tau':>9}{'中位步长':>10}{'p10':>8}{'p90':>8}{'p99':>8}"
        f"{'最长':>8}{'>3x 中位':>10}"
    )
    walk = {}
    for key in MEASURES:
        v = np.array([r.get(key, np.nan) for r in recs], float)
        v = np.where(np.isfinite(v), v, 0.0)
        quanta, acc, run = [], 0.0, 0.0
        for j in range(len(recs)):
            if not np.isfinite(dd[j]):
                continue
            acc += v[j]
            run += dd[j]
            if run >= interval:
                quanta.append(acc)
                acc, run = 0.0, 0.0
        if len(quanta) < 20:
            continue
        tau = float(np.median(quanta))
        steps, acc, run = [], 0.0, 0.0
        for j in range(len(recs)):
            if not np.isfinite(dd[j]):
                continue
            acc += v[j]
            run += dd[j]
            if acc >= tau:
                steps.append(run)
                acc, run = 0.0, 0.0
        if len(steps) < 20:
            continue
        s = np.array(steps)
        walk[key] = {"tau": tau, "steps": s.tolist()}
        print(
            f"  {key:<13}{tau:>9.3f}{np.median(s):>10.2f}{np.percentile(s, 10):>8.2f}"
            f"{np.percentile(s, 90):>8.2f}{np.percentile(s, 99):>8.2f}{s.max():>8.2f}"
            f"{100.0 * (s > 3 * np.median(s)).mean():>9.1f}%"
        )
    print(f"  (步长越分散说明自适应越强；p99/最长 = step_max_dist 的下界)")

    print(f"\n  -- 「一直没变」有多常见:连续多少米才累积到一个 tau --")
    for key, w in walk.items():
        s = np.array(w["steps"])
        for cut in (5.0, 10.0, 20.0):
            frac = 100.0 * (s > cut).mean()
            if frac >= 0.5:
                print(
                    f"  {key:<13}{frac:>6.1f}% 的采纳步跨过 {cut:g} m "
                    f"(这些步覆盖了 {100.0 * s[s > cut].sum() / s.sum():.1f}% 的里程)"
                )
    return walk


def _calibrate_budget(med, rate, fraction=0.8):

    wide = [k for k in sorted(med) if k >= 6]
    if not wide or rate <= 0:
        return None, None
    saturation = max(med[k] for k in wide)
    target = fraction * saturation
    ks = sorted(med)
    for i, k in enumerate(ks):
        if med[k] >= target:
            if i == 0:
                return rate * k, saturation
            k0, k1 = ks[i - 1], k
            span = med[k1] - med[k0]
            frac = 0.0 if span <= 0 else (target - med[k0]) / span
            return rate * (k0 + frac * (k1 - k0)), saturation
    return rate * ks[-1], saturation


def _appearance_step_distances(rows, pairs, breaks, tau):

    adopted = []
    acc_change = 0.0
    acc_dist = 0.0
    previous_b = None
    break_set = set(breaks)
    for row in rows:
        key = f"{row['a']}|{row['b']}"
        if previous_b is not None and row["a"] != previous_b:
            acc_change = acc_dist = 0.0
        previous_b = row["b"]
        if key in break_set:
            acc_change = acc_dist = 0.0
            continue
        distance = row.get("dd")
        if distance is None or not np.isfinite(distance):
            continue
        acc_change += float(pairs[key])
        acc_dist += abs(float(distance))
        if acc_change >= tau:
            adopted.append(acc_dist)
            acc_change = acc_dist = 0.0
    return np.asarray(adopted, dtype=float)


def emit_walk(scene, measure):

    tables, med = {}, {}
    for k in (1, 3, 6, 12, 24):
        path = os.path.join(ROOT, "assets", "map_feature_rate", f"{scene}_s{k}.json")
        if not os.path.isfile(path):
            if k <= 3:
                print(f"missing {os.path.relpath(path, ROOT)}; run --stride {k} first")
                return 1
            continue
        with open(path, "r", encoding="utf-8") as fh:
            tables[k] = json.load(fh)
        v = np.array([r.get(measure, np.nan) for r in tables[k]], float)
        med[k] = float(np.nanmedian(v))
    rate = (med[3] - med[1]) / 2.0
    offset = max(0.0, med[1] - rate)

    pairs, breaks = {}, []
    for r in tables[1]:
        key = f"{r['a']}|{r['b']}"
        v = r.get(measure)
        if v is None or not np.isfinite(v):
            v = 1.0
        pairs[key] = round(max(0.0, float(v) - offset), 6)
        if not r.get("inliers"):
            breaks.append(key)

    budget, sat = _calibrate_budget(med, rate)
    from core.config import STEP_FEATURE_TAU

    appearance_steps = _appearance_step_distances(
        tables[1],
        pairs,
        breaks,
        STEP_FEATURE_TAU,
    )
    if len(appearance_steps):
        try:
            auto_max = float(np.percentile(appearance_steps, 99, method="higher"))
        except TypeError:
            auto_max = float(
                np.percentile(appearance_steps, 99, interpolation="higher")
            )
    else:
        auto_max = None
    broken = []
    if len(pairs) and len(breaks) > 0.5 * len(pairs):
        broken.append(
            f"{len(breaks)}/{len(pairs)} 对被标成断点(>50%)"
            " —— 建图相机连相邻帧都匹配不上,多半是某次测量在坏环境跑的"
        )
    if rate <= 0:
        broken.append(f"单步变化率 rate={rate:.4f} <= 0 —— 各 stride 的测量互相矛盾")
    if sat is None or budget is None:
        broken.append("饱和拟合失败(saturation/budget_half 为 None)")
    if auto_max is None or len(appearance_steps) < 20:
        broken.append(
            f"appearance-only step_max calibration has only {len(appearance_steps)} steps"
        )
    if broken:
        raise SystemExit(
            f"[{scene}] 拒绝写 walk 表:\n  - "
            + "\n  - ".join(broken)
            + f"\n  删掉 assets/map_feature_rate/{scene}_s*.json 后在正确环境里重测。"
        )
    out = {
        "scene": scene,
        "measure": measure,
        "offset": round(offset, 6),
        "rate": round(rate, 6),
        "medians": {str(k): round(v, 6) for k, v in sorted(med.items())},
        "saturation": round(sat, 6),
        "budget_half": round(budget, 6),
        "step_feature_tau": STEP_FEATURE_TAU,
        "step_max_dist_auto_p99": round(auto_max, 6),
        "step_distance_stats_m": {
            "n": int(len(appearance_steps)),
            "median": round(float(np.median(appearance_steps)), 6),
            "p90": round(float(np.percentile(appearance_steps, 90)), 6),
            "p99_higher": round(auto_max, 6),
            "max": round(float(np.max(appearance_steps)), 6),
        },
        "n_pairs": len(pairs),
        "n_breaks": len(breaks),
        "pairs": pairs,
        "breaks": breaks,
    }
    path = os.path.join(ROOT, "assets", "map_feature_rate", f"{scene}_walk.json")
    with open(path + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(out, fh)
    os.replace(path + ".tmp", path)
    print(
        f"[{scene}] {measure}: offset {offset:.4f} + rate {rate:.4f}/frame "
        f"(medians {med[1]:.4f} @1, {med[3]:.4f} @3) -> "
        f"{os.path.relpath(path, ROOT)} ({len(pairs)} pairs, "
        f"{len(breaks)} trajectory breaks)"
    )
    corrected = np.array(list(pairs.values()))
    print(
        f"  corrected per-pair change: median {np.median(corrected):.4f}, "
        f"p90 {np.percentile(corrected, 90):.4f}, "
        f"{100.0 * (corrected <= 0).mean():.1f}% at zero"
    )
    if budget is None:
        print(
            "  budget: NOT calibrated -- run --stride 6/12/24 to measure where "
            "the direct change saturates, otherwise step_budget is a guess"
        )
    else:
        print(
            f"  direct change saturates at {sat:.3f}; 80% of that is reached "
            f"after ~{budget / rate:.1f} frames -> step_budget {budget:.3f} "
            f"(chained units, which is what the walk accumulates)"
        )
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--model-id", default="MegaLoc_8448")
    ap.add_argument("--limit", type=int, default=0, help="cap pairs (0 = whole map)")
    ap.add_argument("--stride", type=int, default=1, help="pair frame i with i+stride")
    ap.add_argument(
        "--report", action="store_true", help="reuse the cache, do not match"
    )
    ap.add_argument(
        "--check",
        action="store_true",
        help="exit 0 when a non-empty stride JSON is present",
    )
    ap.add_argument(
        "--emit-walk",
        action="store_true",
        help="fit the per-pair offset from the stride-1/3 tables and write "
        "<Scene>_walk.json, the change table the walk consumes",
    )
    ap.add_argument(
        "--measure",
        default="view_change",
        choices=MEASURES,
        help="which change measure the walk table carries",
    )
    a = ap.parse_args()

    os.environ.setdefault("DATASET", a.scene.lower())
    out = os.path.join(
        ROOT, "assets", "map_feature_rate", f"{a.scene}_s{a.stride}.json"
    )
    if a.check:
        ok, why = _measurement_status(a.scene, a.stride, out)
        print(f"[{a.scene}] stride {a.stride}: {why}")
        return 0 if ok else 10
    if a.emit_walk:
        return emit_walk(a.scene, a.measure)

    if a.report:
        ok, why = _measurement_status(a.scene, a.stride, out)
        if not ok:
            print(f"cannot reuse {os.path.relpath(out, ROOT)}: {why}")
            return 10
        with open(out, "r", encoding="utf-8") as fh:
            recs = json.load(fh)
    else:
        recs = measure(a.scene, a.model_id, a.limit, a.stride)
        if not recs:
            print("no usable pairs")
            return 1
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out + ".tmp", "w", encoding="utf-8") as fh:
            json.dump(recs, fh)
        os.replace(out + ".tmp", out)
        print(f"  -> {os.path.relpath(out, ROOT)}")

    report(a.scene, recs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
