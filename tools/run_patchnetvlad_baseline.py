import os
import re
import sys
import json
import argparse
import configparser
import hashlib
from collections import defaultdict

import numpy as np
import torch
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO)
sys.path.append(os.path.join(REPO, "3rdparty", "Patch-NetVLAD"))
os.chdir(REPO)

from patchnetvlad.models.models_generic import get_backend, get_model, get_pca_encoding
from patchnetvlad.models.local_matcher import (
    calc_keypoint_centers_from_patches,
    normalise_func,
)
from patchnetvlad.tools.patch_matcher import PatchMatcher
from patchnetvlad.tools.datasets import input_transform

PNV_ROOT = os.path.join(REPO, "3rdparty", "Patch-NetVLAD")
CKPT = os.path.join(
    PNV_ROOT, "patchnetvlad", "pretrained_models", "mapillary_WPCA4096.pth.tar"
)
from core.paper_config import load_paper_config

_SCENE_THRESHOLDS = load_paper_config()["scene_thresholds"]


def _radii(scene):

    base = re.sub(r"C(a\d{4}|full|mix)$", "", scene)
    own = float(_SCENE_THRESHOLDS[scene if scene in _SCENE_THRESHOLDS else base])
    return own, sorted({own, 10.0, 15.0})


RESULT_DIR = os.path.join(REPO, "assets", "paper_baselines")
from core.baseline_protocols import PATCHNETVLAD_PROTOCOL as PROTOCOL_VERSION


def _trajectory_cache(scene):
    return os.path.join(
        REPO,
        "assets",
        "MegaLoc_8448",
        f"test_trajectories_{scene}_no_roi.json",
    )


def _trajectory_fingerprint(scene):
    from sl_agent.cache_fingerprint import trajectory_fingerprint

    return trajectory_fingerprint(scene, protocol=PROTOCOL_VERSION)


def _source_fingerprint(scene):

    digest = hashlib.sha256()
    digest.update(_trajectory_fingerprint(scene).encode("ascii"))
    with open(CKPT, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _result_path(scene):
    return os.path.join(RESULT_DIR, f"patchnetvlad_{scene}_no_roi.json")


def _load_reusable_result(scene, fingerprint):
    path = _result_path(scene)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            result = json.load(handle)
        if result.get("source_fingerprint") != fingerprint:
            return None
        if not isinstance(result.get("values"), dict):
            return None
        if any(str(float(t)) not in result["values"] for t in _radii(scene)[1]):
            return None
        return result
    except (OSError, ValueError):
        return None


def _save_result(result):
    os.makedirs(RESULT_DIR, exist_ok=True)
    path = _result_path(result["scene"])
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    os.replace(tmp, path)


def build_model(config, device):
    encoder_dim, encoder = get_backend()
    checkpoint = torch.load(CKPT, map_location="cpu", weights_only=False)
    assert checkpoint["state_dict"]["WPCA.0.bias"].shape[0] == int(
        config["global_params"]["num_pcs"]
    )
    config["global_params"]["num_clusters"] = str(
        checkpoint["state_dict"]["pool.centroids"].shape[0]
    )
    model = get_model(
        encoder, encoder_dim, config["global_params"], append_pca_layer=True
    )
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def extract_patch_feats(model, img_path, tf, device, pool_size):
    if isinstance(img_path, Image.Image):
        img = img_path
    else:
        img = Image.open(img_path).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)
    enc = model.encoder(x)
    vlad_local, _ = model.pool(enc)
    feats = []
    for this_local in vlad_local:
        pca = (
            get_pca_encoding(
                model, this_local.permute(2, 0, 1).reshape(-1, this_local.size(1))
            )
            .reshape(this_local.size(2), this_local.size(0), pool_size)
            .permute(1, 2, 0)
        )
        feats.append(pca[0])
    return feats


def resolve_paths(scene):

    import core.vpr_core as vc

    q_stem2path = {}
    for p in vc.get_all_images_recursive(f"datasets/{scene}/queries_test"):
        stem = os.path.basename(p).split(".")[0]
        q_stem2path.setdefault(stem, p)
    db_name2path = {}
    for p in vc.get_all_images_recursive(f"datasets/{scene}/database"):
        db_name2path.setdefault(os.path.basename(p), p)
    return q_stem2path, db_name2path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--scenes",
        default="RobotCar,BusinessCampus,Neighborhood,OldTown,NCLT,CustomCampus,Boreas",
    )
    ap.add_argument(
        "--n-queries",
        type=int,
        default=0,
        help="diagnostic subsample size; 0 (default) evaluates the full test split",
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    device = torch.device(a.device)

    scenes = [scene.strip() for scene in a.scenes.split(",") if scene.strip()]
    fingerprints = {scene: _source_fingerprint(scene) for scene in scenes}
    pending = []
    if a.n_queries <= 0:
        for scene in scenes:
            result = _load_reusable_result(scene, fingerprints[scene])
            if result is None:
                pending.append(scene)
            else:
                print(
                    f"[skip Patch-NetVLAD] {scene}: reusable full-test result "
                    f"{result['hits']}/{result['queries']}"
                )
        if not pending:
            return
    else:
        pending = scenes

    config = configparser.ConfigParser()
    config.read(os.path.join(PNV_ROOT, "patchnetvlad", "configs", "performance.ini"))
    pool_size = int(config["global_params"]["num_pcs"])
    patch_sizes = [int(s) for s in config["global_params"]["patch_sizes"].split(",")]
    strides = [int(s) for s in config["global_params"]["strides"].split(",")]
    patch_weights = np.array(
        config["feature_match"]["patchWeights2Use"].split(",")
    ).astype(float)

    all_kps, all_idx = [], []
    for ps, st in zip(patch_sizes, strides):
        kps, idx = calc_keypoint_centers_from_patches(
            config["feature_match"], ps, ps, st, st
        )
        all_kps.append(kps)
        all_idx.append(idx)
    matcher = PatchMatcher(
        config["feature_match"]["matcher"], patch_sizes, strides, all_kps, all_idx
    )

    model = build_model(config, device)
    tf = input_transform(
        resize=(
            int(config["feature_extract"]["imageresizeH"]),
            int(config["feature_extract"]["imageresizeW"]),
        )
    )
    print(
        f"[PatchNetVLAD] ckpt={os.path.basename(CKPT)} device={device} "
        f"patch_sizes={patch_sizes}"
    )

    for scene in pending:
        th, all_th = _radii(scene)
        q_stem2path, db_name2path = resolve_paths(scene)
        qd = defaultdict(list)
        with open(_trajectory_cache(scene), "r", encoding="utf-8") as handle:
            for it in json.load(handle):
                qd[it["query"]].append(it)
        queries = sorted(qd.keys())
        if a.n_queries > 0 and len(queries) > a.n_queries:
            step = max(1, len(queries) // a.n_queries)
            queries = queries[::step][: a.n_queries]

        n = 0
        raw_hits_by_th = {t: 0 for t in all_th}
        pnv_hits_by_th = {t: 0 for t in all_th}
        for q in queries:
            if q not in q_stem2path:
                continue
            cands = sorted(qd[q], key=lambda x: x.get("rank", 999))[:10]
            cpaths = [db_name2path.get(c["candidate_name"]) for c in cands]
            if len(cands) < 2 or any(p is None for p in cpaths):
                continue
            errs = [c.get("error", -1.0) for c in cands]
            ok_by_th = {t: [0 <= e <= t for e in errs] for t in all_th}
            n += 1
            for t in all_th:
                raw_hits_by_th[t] += ok_by_th[t][0]

            qfeat = [
                torch.transpose(f, 0, 1)
                for f in extract_patch_feats(
                    model, q_stem2path[q], tf, device, pool_size
                )
            ]
            diffs = np.zeros((len(cands), len(patch_sizes)))
            for k, cp in enumerate(cpaths):
                dbfeat = extract_patch_feats(model, cp, tf, device, pool_size)
                diffs[k, :], _, _ = matcher.match(qfeat, dbfeat)
            scores = normalise_func(diffs, len(patch_sizes), patch_weights)
            best = int(np.argsort(scores)[0])
            for t in all_th:
                pnv_hits_by_th[t] += ok_by_th[t][best]

        if not n:
            raise RuntimeError(
                f"Patch-NetVLAD evaluated zero usable queries for {scene}"
            )
        values = {str(float(t)): 100.0 * pnv_hits_by_th[t] / n for t in all_th}
        raw_values = {str(float(t)): 100.0 * raw_hits_by_th[t] / n for t in all_th}
        pnv_hit, raw_hit = pnv_hits_by_th[th], raw_hits_by_th[th]
        value = values[str(float(th))]
        print(f"PNV RESULT {scene} over {n} queries")
        for t in all_th:
            print(
                f"  @{t:g}m  Raw {raw_values[str(float(t))]:.1f}%"
                f"  ->  +Patch-NetVLAD {values[str(float(t))]:.1f}%"
            )
        if a.n_queries <= 0:
            _save_result(
                {
                    "schema_version": 1,
                    "protocol_version": PROTOCOL_VERSION,
                    "method": "Patch-NetVLAD",
                    "scene": scene,
                    "threshold_m": th,
                    "hits": int(pnv_hit),
                    "queries": int(n),
                    "value": value,
                    "values": values,
                    "raw_hits": int(raw_hit),
                    "raw_values": raw_values,
                    "trajectory_fingerprint": _trajectory_fingerprint(scene),
                    "source_fingerprint": fingerprints[scene],
                }
            )


if __name__ == "__main__":
    main()
