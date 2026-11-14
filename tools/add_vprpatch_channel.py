import argparse
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

SHORT_EDGE = 322
PATCH = 14
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
EXTS = (".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp")
FIELD, SHUF = "vprp_sim", "vprp_shuf"


def _weights_path():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id="gberton/MegaLoc", filename="model.safetensors")


def provenance():

    from sl_agent.cache_fingerprint import file_sha256

    return {
        "model": "MegaLoc backbone patch tokens",
        "weight_sha256": file_sha256(_weights_path()),
        "short_edge": SHORT_EDGE,
        "score": "mean_max_cosine",
        "frames": "centre",
        "shuffle": "across queries within rank slot, seed=<scene>/<split>/<rank>",
    }


def load_model():
    hub = os.path.expanduser("~/.cache/torch/hub/gmberton_MegaLoc_main")
    _weights_path()
    model = torch.hub.load(hub, "get_trained_model", source="local")
    return model.eval().to(DEV)


def target_size(w, h):
    s = SHORT_EDGE / min(w, h)
    return max(PATCH, round(w * s / PATCH) * PATCH), max(
        PATCH, round(h * s / PATCH) * PATCH
    )


def load_tensor(path):

    im = Image.open(path).convert("RGB")
    W, H = target_size(*im.size)
    arr = np.array(im.resize((W, H), Image.BILINEAR), dtype=np.uint8)
    return torch.from_numpy(arr).permute(2, 0, 1)


def forward_tokens(model, t):

    mean, std = MEAN.to(DEV), STD.to(DEV)
    x = (t[None].to(DEV).float() / 255.0 - mean) / std
    with torch.inference_mode():
        p, _cls = model.backbone(x)
    return F.normalize(p[0].flatten(1).T, dim=1).half().cpu()


class _Images(torch.utils.data.Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return i, load_tensor(self.paths[i])


def embed_iter(model, paths, workers=6):

    loader = torch.utils.data.DataLoader(
        _Images(paths), batch_size=None, num_workers=workers, prefetch_factor=8
    )
    for i, t in loader:
        yield int(i), forward_tokens(model, t)


def score(q, d):

    return float((q.float() @ d.float().T).max(dim=1).values.mean())


def _rel_of_asset(asset_path, asset_dir):
    return os.path.relpath(asset_path, asset_dir)[:-3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--model-id", default="MegaLoc_8448")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument(
        "--limit-queries",
        type=int,
        default=0,
        help="smoke test only: score the first N queries and write NOTHING",
    )
    a = ap.parse_args()

    from core.config import data_tag
    from sl_agent.cache_fingerprint import (
        injector_current,
        read_fingerprint,
        refresh_output_fingerprint,
    )

    caches = {
        s: f"assets/{a.model_id}/{s}_trajectories_{a.scene}{data_tag()}.json"
        for s in a.splits.split(",")
    }
    caches = {s: p for s, p in caches.items() if os.path.exists(p)}
    if not caches:
        raise SystemExit(f"[vprpatch] no cache files for {a.scene} / {a.model_id}")
    prov = provenance()
    loaded = {}
    for s, p in caches.items():
        if (
            not a.force
            and not a.limit_queries
            and injector_current(p, "vprpatch", prov, expected_steps=1)
        ):
            print(f"[vprpatch] {a.scene}/{s}: already current")
            continue
        stamp = read_fingerprint(p) or {}
        if not a.limit_queries and not (
            stamp.get("complete") and stamp.get("semantic_digest")
        ):
            raise SystemExit(
                f"[vprpatch] {p}: state cache has no complete builder fingerprint "
                "-- rebuild it (reproduce.py build stage) before injecting"
            )
        loaded[s] = json.load(open(p, encoding="utf-8"))
    if not loaded:
        return

    asset_dir = f"assets/sequence_assets/{a.scene}"
    index = {}
    for dirpath, _d, files in os.walk(
        os.path.join(asset_dir, "database"), followlinks=True
    ):
        for f in files:
            if f.endswith(".pt"):
                index.setdefault(f[:-3], os.path.join(dirpath, f))
    if not index:
        raise SystemExit(f"[vprpatch] no sequence assets under {asset_dir}/database")
    frame_of = {}
    for rows in loaded.values():
        for r in rows:
            name = r["candidate_name"]
            if name not in frame_of:
                hit = index.get(name)
                frame_of[name] = (
                    os.path.join("datasets", a.scene, _rel_of_asset(hit, asset_dir))
                    if hit
                    else None
                )
    missing = [n for n, f in frame_of.items() if f is None]
    if missing:
        raise SystemExit(
            f"[vprpatch] {len(missing)} candidates have no database frame asset, "
            f"e.g. {missing[:3]}"
        )

    qpath = {}
    for split, rows in loaded.items():
        qdir = f"datasets/{a.scene}/queries_{split}"
        stem = {}
        for dp, _d, fs in os.walk(qdir, followlinks=True):
            for f in fs:
                if f.lower().endswith(EXTS):
                    stem.setdefault(os.path.splitext(f)[0], os.path.join(dp, f))
        want = {str(r["query"]) for r in rows}
        lost = [q for q in want if q not in stem]
        if lost:
            raise SystemExit(
                f"[vprpatch] {a.scene}/{split}: {len(lost)} queries have no image "
                f"under {qdir}, e.g. {lost[:3]}"
            )
        qpath[split] = stem

    if a.limit_queries:
        for split in loaded:
            keep, seen = [], set()
            for r in loaded[split]:
                if r["query"] not in seen and len(seen) >= a.limit_queries:
                    continue
                seen.add(r["query"])
                keep.append(r)
            loaded[split] = keep
        need = sorted(
            {frame_of[r["candidate_name"]] for rows in loaded.values() for r in rows}
        )
    else:
        need = sorted({f for f in frame_of.values()})

    model = load_model()
    sizes = []
    for p in need:
        with Image.open(p) as im:
            W, H = target_size(*im.size)
        sizes.append((W // PATCH) * (H // PATCH))
    offs = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
    tmp_dir = os.path.join("assets", "vprpatch_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    store_path = os.path.join(tmp_dir, f"{a.scene}_{a.model_id}_{os.getpid()}.npy")
    store = np.lib.format.open_memmap(
        store_path, mode="w+", dtype=np.float16, shape=(int(offs[-1]), 768)
    )
    print(
        f"[vprpatch] {a.scene}: embedding {len(need)} database frames ...", flush=True
    )
    try:
        for n, (i, feat) in enumerate(embed_iter(model, need, a.workers)):
            if feat.shape[0] != sizes[i]:
                raise SystemExit(f"patch count mismatch for {need[i]}")
            store[offs[i] : offs[i + 1]] = feat.numpy()
            if n % 2000 == 0:
                print(f"    db {n}/{len(need)}", flush=True)
        store.flush()
        db_row = {p: i for i, p in enumerate(need)}

        for split, rows in loaded.items():
            by_q = {}
            for r in rows:
                by_q.setdefault(str(r["query"]), []).append(r)
            qnames = sorted(by_q)
            paths = [qpath[split][q] for q in qnames]
            print(
                f"[vprpatch] {a.scene}/{split}: {len(qnames)} queries, {len(rows)} rows",
                flush=True,
            )
            for n, (i, qf) in enumerate(embed_iter(model, paths, a.workers)):
                qg = qf.to(DEV)
                for r in by_q[qnames[i]]:
                    k = db_row[frame_of[r["candidate_name"]]]
                    d = torch.from_numpy(np.asarray(store[offs[k] : offs[k + 1]])).to(
                        DEV
                    )
                    r[FIELD] = [round(score(qg, d), 5)]
                if n % 1000 == 0:
                    print(f"    {split} {n}/{len(qnames)}", flush=True)
            slots = {}
            for r in rows:
                slots.setdefault(int(r.get("rank", 0)), []).append(r)
            for rank, members in sorted(slots.items()):
                vals = [m[FIELD][0] for m in members]
                random.Random(f"{a.scene}/{split}/{rank}").shuffle(vals)
                for m, v in zip(members, vals):
                    m[SHUF] = [v]
            bad = [r for r in rows if len(r.get(FIELD) or []) != 1]
            if bad:
                raise SystemExit(
                    f"[vprpatch] {a.scene}/{split}: {len(bad)} rows unscored"
                )
            if a.limit_queries:
                v = np.array([r[FIELD][0] for r in rows])
                print(
                    f"[vprpatch] smoke {split}: {len(rows)} rows, vprp_sim "
                    f"min {v.min():.3f} median {np.median(v):.3f} max {v.max():.3f} (NOT written)"
                )
                continue
            tmp = caches[split] + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(rows, fh)
            os.replace(tmp, caches[split])
            refresh_output_fingerprint(caches[split], vprpatch=prov)
            print(
                f"[vprpatch] {a.scene}/{split}: wrote {len(rows)} rows -> {caches[split]}"
            )
    finally:
        del store
        if os.path.exists(store_path):
            os.remove(store_path)


if __name__ == "__main__":
    main()
