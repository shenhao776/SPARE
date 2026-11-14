#!/usr/bin/env python

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PITTS_ROOT = os.path.expanduser("~/datasets/Pittsburgh250k")
PITTS_OVERLAY = "datasets/public_benchmarks/pitts30k_overlay"
PITTS_STAGED = "datasets/public_benchmarks/pitts30k"

MSLS_ROOT = "datasets/public_benchmarks/msls"
MSLS_SPLIT = "3rdparty/Pair-VPR/pairvpr/datasets/datasetfiles/msls_val"


def _resolve(name, root, sub=""):

    for base in (PITTS_STAGED, PITTS_OVERLAY, root):
        p = os.path.join(base, sub, name) if sub else os.path.join(base, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"找不到 {name}(overlay 与 {root} 都没有)")


def load_pitts30k():
    from scipy.io import loadmat

    mat = os.path.join(PITTS_ROOT, "netvlad_splits/datasets/pitts30k_test.mat")
    db = loadmat(mat, struct_as_record=False, squeeze_me=True)["dbStruct"]
    db_paths = [_resolve(str(x), PITTS_ROOT) for x in db.dbImageFns]
    q_paths = [_resolve(str(x), PITTS_ROOT, "queries_real") for x in db.qImageFns]
    utm_db = np.asarray(db.utmDb, dtype=np.float64)
    utm_q = np.asarray(db.utmQ, dtype=np.float64)
    if utm_db.shape[0] == 2:
        utm_db = utm_db.T
    if utm_q.shape[0] == 2:
        utm_q = utm_q.T
    return db_paths, q_paths, utm_db, utm_q, float(db.posDistThr)


def _msls_coords():

    import csv as _csv

    coord = {}
    for city in ("cph", "sf"):
        for part in ("database", "query"):
            f = os.path.join(MSLS_ROOT, "train_val", city, part, "postprocessed.csv")
            for r in _csv.DictReader(open(f, encoding="utf-8")):
                coord[r["key"]] = (float(r["easting"]), float(r["northing"]))
    return coord


def load_msls_val():

    import numpy as np

    d = MSLS_SPLIT
    db = np.load(os.path.join(d, "msls_val_dbImages.npy"))
    q = np.load(os.path.join(d, "msls_val_qImages.npy"))
    q_idx = np.load(os.path.join(d, "msls_val_qIdx.npy"))
    p_idx = np.load(os.path.join(d, "msls_val_pIdx.npy"), allow_pickle=True)
    db_paths = [os.path.join(MSLS_ROOT, str(x)) for x in db]
    q_paths = [os.path.join(MSLS_ROOT, str(q[i])) for i in q_idx]
    gt = [set(int(v) for v in np.atleast_1d(p)) for p in p_idx]
    missing = [p for p in (db_paths[:1] + q_paths[:1]) if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"MSLS 图像不在 {MSLS_ROOT}:{missing[0]}")
    coord = _msls_coords()
    k = lambda p: os.path.splitext(os.path.basename(str(p)))[0]
    utm_db = np.array([coord[k(x)] for x in db])
    utm_q = np.array([coord[k(q[i])] for i in q_idx])
    return db_paths, q_paths, gt, utm_db, utm_q


def recalls_by_index(desc_db, desc_q, gt, ks=(1, 5, 10)):

    import numpy as np

    kmax = max(ks)
    hits = {k: 0 for k in ks}
    step = 512
    for s0 in range(0, len(desc_q), step):
        sim = desc_q[s0 : s0 + step] @ desc_db.T
        idx = np.argpartition(-sim, kmax, axis=1)[:, :kmax]
        ordr = np.take_along_axis(sim, idx, 1).argsort(axis=1)[:, ::-1]
        idx = np.take_along_axis(idx, ordr, 1)
        for r, row in enumerate(idx):
            pos = gt[s0 + r]
            for k in ks:
                if any(int(c) in pos for c in row[:k]):
                    hits[k] += 1
    n = len(desc_q)
    return {str(k): round(100.0 * hits[k] / n, 1) for k in ks}


def encode(model, paths, device, batch_size, workers, transform):
    from core.vpr_core import SimpleImageDataset

    loader = DataLoader(
        SimpleImageDataset(paths, transform),
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=True,
    )
    out = []
    with torch.no_grad():
        for i, (imgs, _) in enumerate(loader):
            d = model(imgs.to(device))
            out.append(torch.nn.functional.normalize(d, p=2, dim=1).cpu())
            if (i + 1) % 20 == 0:
                print(f"    {(i+1)*batch_size}/{len(paths)}", flush=True)
    return torch.cat(out).numpy().astype(np.float32)


def recalls_multi(desc_db, desc_q, utm_db, utm_q, thrs, ks=(1, 5, 10)):

    return {f"{t:g}": recalls(desc_db, desc_q, utm_db, utm_q, t, ks) for t in thrs}


def oracle_bound(utm_db, utm_q, thrs):

    import numpy as np

    step = 512
    nn = []
    for s in range(0, len(utm_q), step):
        d = np.linalg.norm(utm_db[None, :, :] - utm_q[s : s + step, None, :], axis=2)
        nn.append(d.min(axis=1))
    nn = np.concatenate(nn)
    return {f"{t:g}": round(100.0 * float((nn <= t).mean()), 1) for t in thrs}


def recalls(desc_db, desc_q, utm_db, utm_q, thr, ks=(1, 5, 10)):

    kmax = max(ks)
    hits = {k: 0 for k in ks}
    step = 512
    for s in range(0, len(desc_q), step):
        q = desc_q[s : s + step]
        sim = q @ desc_db.T
        idx = np.argpartition(-sim, kmax, axis=1)[:, :kmax]
        ordr = np.take_along_axis(sim, idx, 1).argsort(axis=1)[:, ::-1]
        idx = np.take_along_axis(idx, ordr, 1)
        d = np.linalg.norm(utm_db[idx] - utm_q[s : s + step, None, :], axis=2)
        ok = d <= thr
        for k in ks:
            hits[k] += int(ok[:, :k].any(axis=1).sum())
    n = len(desc_q)
    return {str(k): round(100.0 * hits[k] / n, 1) for k in ks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="pitts30k", choices=["pitts30k", "msls_val"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--radii",
        nargs="+",
        type=float,
        default=[2, 5, 10, 15, 25],
        help="pitts30k:一次编码算多个判定半径",
    )
    ap.add_argument(
        "--resize",
        nargs=2,
        type=int,
        default=None,
        metavar=("H", "W"),
        help="先统一尺寸再走模型自己的预处理",
    )
    ap.add_argument(
        "--save-desc", default=None, help="write descriptors + GT to this .npz"
    )
    ap.add_argument(
        "--load-desc", default=None, help="evaluate from a --save-desc .npz"
    )
    a = ap.parse_args()

    if a.load_desc:
        return _eval_from_desc(a)

    import core.config as CFG_MOD
    from core.models import build_model, get_transform

    name = CFG_MOD.VPR_MODEL_TYPE
    a.out = a.out or f"assets/public_benchmarks/{a.benchmark}.json"
    if a.benchmark == "pitts30k":
        db_paths, q_paths, utm_db, utm_q, thr = load_pitts30k()
        gt = None
        print(f"[{a.benchmark}] db={len(db_paths)} q={len(q_paths)} GT 半径={thr:g} m")
    else:
        db_paths, q_paths, gt, utm_db, utm_q = load_msls_val()
        thr = 25.0
        print(
            f"[{a.benchmark}] db={len(db_paths)} q={len(q_paths)} "
            f"官方 posDistThr={thr:g} m(evaluate.py 默认,无朝向阈值)"
        )

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    results = json.load(open(a.out)) if os.path.exists(a.out) else {}
    if name in results and not a.force:
        print(f"[skip] {name} 已有结果:{results[name]['recall']}")
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(
        f"\n=== {name} (dim {CFG_MOD.CFG.model_dict.get('feature_dim')}) ===",
        flush=True,
    )
    model = build_model(CFG_MOD.CFG.model_dict, device).eval()
    tf = get_transform(CFG_MOD.CFG.model_dict)
    rs = a.resize or ((480, 640) if a.benchmark == "msls_val" else None)
    if rs:
        from torchvision import transforms as _T

        tf = _T.Compose([_T.Resize(tuple(rs)), tf])
        print(f"  统一尺寸到 {rs[0]}x{rs[1]} 后再走模型预处理")
    print("  encoding database ...", flush=True)
    dd = encode(model, db_paths, device, a.batch_size, a.workers, tf)
    print("  encoding queries ...", flush=True)
    dq = encode(model, q_paths, device, a.batch_size, a.workers, tf)
    if a.save_desc:
        _save_desc(a.save_desc, a.benchmark, name, dd, dq, utm_db, utm_q, gt, thr)
    if gt is None:
        thrs = a.radii
        rmulti = recalls_multi(dd, dq, utm_db, utm_q, thrs)
        r = rmulti[f"{thr:g}"]
        results[name] = {
            "dim": int(dd.shape[1]),
            "thr_m": thr,
            "recall": r,
            "recall_by_radius": rmulti,
            "oracle_by_radius": oracle_bound(utm_db, utm_q, thrs),
            "n_db": len(db_paths),
            "n_q": len(q_paths),
        }
        print(
            "  逐半径 R@1: " + "  ".join(f"@{k}m {v['1']}" for k, v in rmulti.items())
        )
    else:
        r = r_idx = recalls_by_index(dd, dq, gt)
        rmulti = recalls_multi(dd, dq, utm_db, utm_q, a.radii)
        gap = abs(rmulti["25"]["1"] - r_idx["1"])
        flag = "" if gap <= 1.0 else f"  ⚠️ 与索引 GT 差 {gap:.1f} 点,重建口径存疑"
        print(
            f"  交叉验证 @25m: 索引 GT {r_idx['1']}  vs  距离 GT "
            f"{rmulti['25']['1']}{flag}"
        )
        results[name] = {
            "dim": int(dd.shape[1]),
            "thr_m": thr,
            "recall": r_idx,
            "recall_by_radius": rmulti,
            "oracle_by_radius": oracle_bound(utm_db, utm_q, a.radii),
            "n_db": len(db_paths),
            "n_q": len(q_paths),
        }
        print(
            "  逐半径 R@1: " + "  ".join(f"@{k}m {v['1']}" for k, v in rmulti.items())
        )
    json.dump(results, open(a.out, "w"), indent=2, ensure_ascii=False)
    print(f"  R@1={r['1']}  R@5={r['5']}  R@10={r['10']}  (dim {dd.shape[1]})")

    print(f"\n写入 {a.out}")


def _save_desc(path, benchmark, name, dd, dq, utm_db, utm_q, gt, thr):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    extra = {}
    if gt is not None:
        flat = [sorted(g) for g in gt]
        extra["gt_flat"] = np.array([v for g in flat for v in g], dtype=np.int64)
        extra["gt_offsets"] = np.cumsum([0] + [len(g) for g in flat]).astype(np.int64)
    np.savez(
        path,
        benchmark=benchmark,
        model=name,
        thr_m=thr,
        desc_db=dd.astype(np.float16),
        desc_q=dq.astype(np.float16),
        utm_db=utm_db,
        utm_q=utm_q,
        **extra,
    )
    print(f"  descriptors -> {path}")


def _eval_from_desc(a):

    z = np.load(a.load_desc)
    benchmark, name = str(z["benchmark"]), str(z["model"])
    if benchmark != a.benchmark:
        raise SystemExit(f"{a.load_desc} holds {benchmark}, not {a.benchmark}")
    dd = z["desc_db"].astype(np.float32)
    dq = z["desc_q"].astype(np.float32)
    utm_db, utm_q, thr = z["utm_db"], z["utm_q"], float(z["thr_m"])
    print(f"[{benchmark}] {name} from {a.load_desc}: db={len(dd)} q={len(dq)}")
    rmulti = recalls_multi(dd, dq, utm_db, utm_q, a.radii)
    if "gt_flat" in z.files:
        flat, off = z["gt_flat"], z["gt_offsets"]
        gt = [set(int(v) for v in flat[off[i] : off[i + 1]]) for i in range(len(dq))]
        r = recalls_by_index(dd, dq, gt)
    else:
        r = rmulti[f"{thr:g}"]
    result = {
        name: {
            "dim": int(dd.shape[1]),
            "thr_m": thr,
            "recall": r,
            "recall_by_radius": rmulti,
            "oracle_by_radius": oracle_bound(utm_db, utm_q, a.radii),
            "n_db": len(dd),
            "n_q": len(dq),
        }
    }
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(result, open(a.out, "w"), indent=2, ensure_ascii=False)
    print("  R@1 by radius: " + "  ".join(f"@{k}m {v['1']}" for k, v in rmulti.items()))
    return result


if __name__ == "__main__":
    main()
