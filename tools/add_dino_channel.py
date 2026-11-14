import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

IMG_W, IMG_H = 448, 336
MODEL_NAME = "dinov2_vitb14"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _dino_provenance():

    from sl_agent.cache_fingerprint import file_sha256

    hub = os.path.expanduser("~/.cache/torch/hub")
    weight = os.path.join(hub, "checkpoints", "dinov2_vitb14_pretrain.pth")
    return {
        "model_name": MODEL_NAME,
        "image_size": [IMG_W, IMG_H],
        "weight_sha256": file_sha256(weight) if os.path.isfile(weight) else "missing",
    }


def load_model():
    hub = os.path.expanduser("~/.cache/torch/hub")
    torch.hub.set_dir(hub)
    local_repo = os.path.join(hub, "facebookresearch_dinov2_main")
    if os.path.isdir(local_repo):
        m = torch.hub.load(local_repo, MODEL_NAME, source="local", pretrained=True)
    else:
        m = torch.hub.load(
            "facebookresearch/dinov2", MODEL_NAME, source="github", pretrained=True
        )
    return m.eval().to(DEV)


def embed(model, paths, out_path, batch=16):

    mean, std = MEAN.to(DEV), STD.to(DEV)
    n = len(paths)
    store = None
    index = {}
    for start in range(0, n, batch):
        chunk = paths[start : start + batch]
        tensors = []
        keep = []
        for p in chunk:
            im = cv2.imread(p)
            if im is None:
                continue
            im = cv2.cvtColor(cv2.resize(im, (IMG_W, IMG_H)), cv2.COLOR_BGR2RGB)
            tensors.append(torch.from_numpy(im).float().permute(2, 0, 1))
            keep.append(p)
        if not tensors:
            continue
        t = (torch.stack(tensors).to(DEV) / 255.0 - mean) / std
        with torch.no_grad():
            feats = model.forward_features(t)["x_norm_patchtokens"]
        feats = F.normalize(feats, dim=-1).half().cpu().numpy()
        if store is None:
            store = np.lib.format.open_memmap(
                out_path,
                mode="w+",
                dtype=np.float16,
                shape=(n, feats.shape[1], feats.shape[2]),
            )
        for i, p in enumerate(keep):
            index[p] = start + i
            store[start + i] = feats[i]
        if start % (batch * 20) == 0:
            print(f"    {start + len(keep)}/{n}", flush=True)
    if store is not None:
        store.flush()
    return index


def _rel_of_asset(asset_path, asset_dir):

    return os.path.relpath(asset_path, asset_dir)[:-3]


def _expected_steps():

    from core.config import DINO_K, SPATIAL_STEPS

    return min(DINO_K, sum(SPATIAL_STEPS) + 1)


def _center_slice(window):

    from core.config import DINO_K

    if not window or len(window) <= DINO_K:
        return window
    start = (len(window) - DINO_K) // 2
    return window[start : start + DINO_K]


def patch_score(q, d):

    return float((q @ d.T).max(axis=1).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--model-id", default="MegaLoc_8448")
    ap.add_argument(
        "--force",
        action="store_true",
        help="recompute even if the cache already carries dino_sim",
    )
    a = ap.parse_args()

    asset_dir = f"assets/sequence_assets/{a.scene}"
    if not os.path.isdir(asset_dir):
        raise SystemExit(f"missing sequence assets: {asset_dir}")

    from core.config import SPATIAL_STEPS, data_tag

    caches = {
        s: f"assets/{a.model_id}/{s}_trajectories_{a.scene}{data_tag()}.json"
        for s in a.splits.split(",")
    }
    caches = {s: p for s, p in caches.items() if os.path.exists(p)}
    if not caches:
        raise SystemExit("no cache files found")

    from sl_agent.cache_fingerprint import injector_current

    dino_provenance = _dino_provenance()
    loaded = {}
    for s, p in caches.items():
        if (
            injector_current(
                p, "dino", dino_provenance, expected_steps=_expected_steps()
            )
            and not a.force
        ):
            print(f"[dino] {s}: channel already current; skipping JSON load")
            continue
        rows = json.load(open(p, encoding="utf-8"))
        loaded[s] = rows
    if not loaded:
        return

    index = {}
    dupes = 0
    for dirpath, _dirs, files in os.walk(
        os.path.join(asset_dir, "database"), followlinks=True
    ):
        for f in files:
            if not f.endswith(".pt"):
                continue
            key = f[:-3]
            if key in index:
                dupes += 1
                continue
            index[key] = os.path.join(dirpath, f)
    if not index:
        raise SystemExit(f"{asset_dir}/database 下一个 .pt 都没有")
    if dupes:
        print(f"[dino] warning: {dupes} 个重名资产,保留先遇到的")
    print(f"[dino] 索引了 {len(index)} 个数据库帧特征")

    from core.config import DATA_ABLATION_RANDOM_WINDOW

    windows = {}
    row_windows = {}
    needed = set()
    if DATA_ABLATION_RANDOM_WINDOW:
        from sl_agent.random_window_replay import load_db_rel_paths, replay

        db_list, cand_by_name = load_db_rel_paths(a.model_id, a.scene)
        for split, rows in loaded.items():
            for i, r in enumerate(rows):
                if "step_offset_m" not in r:
                    raise SystemExit(
                        f"[{a.scene}/{split}] 随机窗口缓存还没跑过 inject_step_meta"
                        f"(行里没有 step_offset_m)—— 先跑它:它会用 step_errors "
                        f"逐位校验重放,校验过的帧序列 dino 才能安全复用"
                    )
                rel_p = cand_by_name.get(r["candidate_name"])
                if rel_p is None:
                    row_windows[(split, i)] = None
                    continue
                seq_len = len(r.get("step_errors") or []) or (sum(SPATIAL_STEPS) + 1)
                w = [
                    os.path.join("datasets", a.scene, rp)
                    for rp in replay(a.scene, db_list, r["query"], rel_p, seq_len)
                ]
                w = _center_slice(w)
                row_windows[(split, i)] = w
                needed.update(w)
    else:
        from core.config import SPATIAL_STEPS
        from core.vpr_core import (
            get_database_spatial_sequence,
            pose_manager as _pose_manager,
        )

        for rows in loaded.values():
            for r in rows:
                name = r["candidate_name"]
                if name not in windows:
                    hit = index.get(name)
                    if hit is None:
                        windows[name] = None
                        continue
                    frame = os.path.join(
                        "datasets", a.scene, _rel_of_asset(hit, asset_dir)
                    )
                    try:
                        back, fwd = get_database_spatial_sequence(
                            os.path.abspath(frame), _pose_manager, SPATIAL_STEPS
                        )
                    except Exception:
                        windows[name] = None
                        continue
                    windows[name] = _center_slice(
                        list(reversed(back)) + [os.path.abspath(frame)] + fwd
                    )
                if windows[name]:
                    needed.update(windows[name])
        missing = [n for n, w in windows.items() if not w]
        if missing:
            print(f"[dino] warning: {len(missing)} candidates have no sequence asset")
        lengths = {len(w) for w in windows.values() if w}
        expected = sum(SPATIAL_STEPS) + 1
        if lengths and lengths != {expected}:
            print(
                f"[dino] warning: window lengths {sorted(lengths)}, expected {expected}"
            )
    needed = sorted(needed)
    print(
        f"[dino] {len(windows) or len(row_windows)} candidate windows, "
        f"{len(needed)} distinct database frames"
        + (" (random-window replay)" if DATA_ABLATION_RANDOM_WINDOW else "")
    )

    model = load_model()
    os.makedirs("assets/dino_cache", exist_ok=True)
    db_store = (
        f"assets/dino_cache/{a.scene}_db_patches_rw.npy"
        if DATA_ABLATION_RANDOM_WINDOW
        else f"assets/dino_cache/{a.scene}_db_patches.npy"
    )
    print("[dino] embedding database frames ...")
    db_index = embed(model, needed, db_store)
    db = np.load(db_store, mmap_mode="r")

    failures = []
    for split, rows in loaded.items():
        qdir = f"datasets/{a.scene}/queries_{split}"
        exts = (".png", ".jpg", ".jpeg", ".tiff", ".bmp")
        qpaths = sorted(
            os.path.join(dp, f)
            for dp, _d, fs in os.walk(qdir, followlinks=True)
            for f in fs
            if f.lower().endswith(exts)
        )
        if not qpaths:
            print(f"[dino] {split}: {qdir} 下一张图都没有,跳过")
            continue
        stem = {os.path.splitext(os.path.basename(p))[0]: p for p in qpaths}
        q_store = f"assets/dino_cache/{a.scene}_{split}_q_patches.npy"
        print(f"[dino] {split}: embedding {len(qpaths)} queries ...")
        q_index = embed(model, qpaths, q_store)
        qf = np.load(q_store, mmap_mode="r")

        done = 0
        for _ri, r in enumerate(rows):
            qp = stem.get(r["query"])
            win = (
                row_windows.get((split, _ri))
                if DATA_ABLATION_RANDOM_WINDOW
                else windows.get(r["candidate_name"])
            )
            if qp is None or qp not in q_index or not win:
                r["dino_sim"] = []
                continue
            qv = np.asarray(qf[q_index[qp]], dtype=np.float32)
            seq = []
            for p in win:
                idx = db_index.get(p)
                seq.append(
                    0.0
                    if idx is None
                    else patch_score(qv, np.asarray(db[idx], dtype=np.float32))
                )
            r["dino_sim"] = [round(v, 5) for v in seq]
            done += 1
            if done % 500 == 0:
                print(f"    {done}/{len(rows)}", flush=True)
        if done < len(rows):
            failures.append(
                f"{split}: {len(rows) - done}/{len(rows)} rows have no dino_sim"
            )
            continue
        tmp = caches[split] + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(rows, fh)
        os.replace(tmp, caches[split])
        from sl_agent.cache_fingerprint import refresh_output_fingerprint

        refresh_output_fingerprint(caches[split], dino=dino_provenance)
        print(
            f"[dino] {split}: wrote verified dino_sim for {done}/{len(rows)} rows -> {caches[split]}"
        )

    for store in [db_store] + [
        f"assets/dino_cache/{a.scene}_{split}_q_patches.npy" for split in loaded
    ]:
        if os.path.isfile(store):
            os.remove(store)

    if failures:
        raise SystemExit("[dino] incomplete:\n  " + "\n  ".join(failures))


if __name__ == "__main__":
    main()
