import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default=None, help="默认取 TARGET_SCENE")
    p.add_argument("--dataset", default=None, help="DATASET 环境变量,须在 import 前设")
    p.add_argument("--n", type=int, default=5, help="计时的 query 数(另加 1 次预热)")
    p.add_argument(
        "--match-batch",
        default="8",
        help="逗号分隔;1 = 逐对(原实现),>1 = 批处理。给多个值就是 A/B",
    )
    p.add_argument(
        "--recompute",
        action="store_true",
        help="对照臂:不读预建的 .pt,现场对每个窗口帧提 SuperPoint。"
        "用来量'地图侧不预建要贵多少'",
    )
    p.add_argument("--topk", type=int, default=None, help="默认取协议的 Top-K")
    p.add_argument(
        "--warm-db",
        action="store_true",
        help="先不计时地把全部 query 跑一遍,让 DINOv2 / MegaLoc 的数据库 patch 特征进缓存,"
        "再计时。对应'数据库侧特征在建图时一次性算好'的部署假设,与基线评测"
        "(rerank_eval_common 在计时前先提完数据库特征)同一口径",
    )
    return p.parse_args()


ARGS = _parse_args()
if ARGS.dataset:
    os.environ["DATASET"] = ARGS.dataset

import numpy as np
import torch
import cv2
from PIL import Image

from core.config import TARGET_SCENE, SPATIAL_STEPS, ABLATION_ARCH
from sl_agent.vpr_classifier import head_steps
from core.vpr_core import (
    ONLINE_DEFAULT_ZOOM,
    VPRService,
    get_all_images_recursive,
    get_base_transform,
    pose_manager,
)
from sl_agent.online_state import OnlineStateBuilder


class Timer:

    STAGES = ("pt", "match", "dino", "vprp", "geom")

    def __init__(self, builder):
        self.b = builder
        self.t = dict.fromkeys(self.STAGES, 0.0)
        self.n_pairs = 0
        self._orig = {}
        self._install()

    def _wrap(self, owner, name, stage, count_pairs=False):
        fn = getattr(owner, name)
        self._orig[(owner, name)] = fn

        def wrapped(*a, **kw):
            s = time.perf_counter()
            r = fn(*a, **kw)
            self.t[stage] += time.perf_counter() - s
            if count_pairs:
                self.n_pairs += len(a[1]) if name.endswith("_batch") else 1
            return r

        setattr(owner, name, wrapped)

    def _install(self):
        b = self.b
        self._wrap(b, "_step_features", "pt")
        self._wrap(b, "_dino_patches", "dino")
        self._wrap(b, "_vprp_tokens", "vprp")
        self._wrap(b.cropper, "match_features", "match", count_pairs=True)
        self._wrap(b.cropper, "match_features_batch", "match", count_pairs=True)
        self._wrap(b.cropper, "compute_geometric_metrics", "geom")

    def restore(self):
        for (owner, name), fn in self._orig.items():
            setattr(owner, name, fn)

    def reset(self):
        for k in self.t:
            self.t[k] = 0.0
        self.n_pairs = 0


def _install_recompute(builder):

    import cv2

    def recompute(rel_path):
        img = cv2.imread(os.path.join(builder.dataset_path, rel_path))
        if img is None:
            raise FileNotFoundError(f"读不到窗口帧: {rel_path}")
        return builder.cropper.extract_image(img)

    builder._step_features = recompute


def main():
    scene = ARGS.scene or TARGET_SCENE
    ds = os.path.join("datasets", scene)
    nominal_L = sum(SPATIAL_STEPS) + 1
    head_L = head_steps(ABLATION_ARCH, nominal_L)

    svc = VPRService()
    top_k = ARGS.topk or getattr(svc.args, "top_k", 10)
    db_desc, db_paths = svc._get_or_compute_db_features(ds)
    db_rel = [str(p) for p in db_paths]
    lookup = {r: db_desc[i] for i, r in enumerate(db_rel)}

    import faiss

    index = faiss.IndexFlatL2(svc.args.fc_output_dim)
    index.add(db_desc)

    queries = get_all_images_recursive(os.path.join(ds, "queries_test"))
    if not queries:
        raise SystemExit(f"{ds}/queries_test 里没有图")
    queries = queries[: ARGS.n + 1]
    tr = get_base_transform()

    print(
        f"场景 {scene}  K={top_k}  协议 L={nominal_L} head 窗口 {head_L}  "
        f"query {len(queries)-1} 张(+1 预热)  "
        f"地图特征={'现场重算' if ARGS.recompute else '读预建 .pt'}"
    )
    print(
        f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}"
    )

    for mb in [int(x) for x in ARGS.match_batch.split(",")]:
        builder = OnlineStateBuilder(svc, ds, scene=scene, match_batch=mb)
        if ARGS.recompute:
            _install_recompute(builder)
        timer = Timer(builder)
        totals, splits, pair_counts = [], [], []

        passes = [(qi, qp) for qi, qp in enumerate(queries)]
        if ARGS.warm_db:
            passes = [(-1, qp) for qp in queries] + passes
        for qi, qp in passes:
            with torch.no_grad():
                qd = (
                    svc.model(
                        tr(Image.open(qp).convert("RGB"))
                        .unsqueeze(0)
                        .to(svc.args.device)
                    )
                    .cpu()
                    .numpy()
                )
            _, pred = index.search(qd, top_k)
            cands = [
                {
                    "rel_path": db_rel[i],
                    "rank": k + 1,
                    "vpr_sim": 0.5,
                    "global_pose": (
                        pose_manager.get_pose_info(os.path.join(ds, db_rel[i])) or {}
                    ).get("global"),
                }
                for k, i in enumerate(pred[0])
            ]
            q_cv = cv2.imread(qp)

            timer.reset()
            s = time.perf_counter()
            rows = builder.build_rows(
                q_cv, qp, cands, lookup, qd[0], zoom=ONLINE_DEFAULT_ZOOM
            )
            builder.build_state_tensor(rows)
            total = time.perf_counter() - s

            if qi <= 0:
                continue
            totals.append(total)
            splits.append(dict(timer.t))
            pair_counts.append(timer.n_pairs)

        timer.restore()
        if not totals:
            raise SystemExit("没有计到时的 query")

        pairs = int(statistics.median(pair_counts))
        expected = top_k * head_L
        mean = statistics.mean(totals)
        agg = {k: statistics.mean(sp[k] for sp in splits) for k in Timer.STAGES}
        other = mean - sum(agg.values())

        print(f"\n===== match_batch = {mb} =====")
        print(f"  实际匹配对数  {pairs}   (K*head_L = {expected})")
        for k, label in (
            ("pt", "读 .pt"),
            ("match", "LightGlue"),
            ("geom", "RANSAC/几何"),
            ("dino", "DINOv2"),
            ("vprp", "MegaLoc patch"),
        ):
            print(f"  {label:<12} {agg[k]*1000:8.0f} ms  ({100*agg[k]/mean:4.1f} %)")
        print(f"  {'其余':<12} {other*1000:8.0f} ms  ({100*other/mean:4.1f} %)")
        print(
            f"  {'总计':<12} {mean*1000:8.0f} ms  ± {statistics.pstdev(totals)*1000:.0f}"
        )
        if pairs:
            print(f"  每对          {agg['match']/pairs*1000:8.2f} ms")

        if pairs != expected:
            print(
                f"\n❌ 匹配对数 {pairs} != K*head_L = {expected}。"
                f"\n   窗口没跑满。最常见的原因:按 step 索引去迭代候选自己的 .pt,"
                f"\n   而精简资产的 lightglue_feats 只有 1 项(该帧自己)。",
                file=sys.stderr,
            )
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
