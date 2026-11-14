import argparse
import os
import sys

import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DHE = os.path.join(ROOT, "3rdparty", "DHE-VPR")

sys.path.insert(0, ROOT)

from tools.rerank_eval_common import (
    add_eval_args,
    evaluate_scenes,
    parse_thresholds,
)

IMG_SIZE = 384


def find_weights(scene, threshold):
    archived = os.path.join(
        ROOT, "assets", "dhe_vpr", f"{scene}_th{threshold:g}_hr.pth"
    )
    if os.path.isfile(archived):
        return archived
    raise SystemExit(
        f"没有 {scene} th{threshold:g} 的归档权重。先训练:\n"
        f"  bash tools/run_dhe_target.sh {scene} {threshold:g}"
    )


def build_model(device, hr_weights):

    sys.path.insert(0, DHE)
    import network

    saved_argv = sys.argv
    sys.argv = [
        "train_dhe.py",
        "--datasets_folder",
        os.path.join(ROOT, "assets", "prgs_data"),
    ]
    try:
        import parser as dhe_parser

        args = dhe_parser.parse_arguments()
    finally:
        sys.argv = saved_argv

    fe = network.FeaturesExtractor(args)
    fe_ckpt = os.path.join(ROOT, "assets", "dhe_vpr", "CCT14_msls.pth")
    state = torch.load(fe_ckpt, map_location="cpu", weights_only=False)[
        "model_state_dict"
    ]
    renamed = {}
    for k, v in state.items():
        k = k.replace("module.backbone", "encoder").replace(
            "module.aggregation", "pool"
        )
        renamed[k] = v
    fe.load_state_dict(renamed)

    hr = network.HomographyRegression()
    payload = torch.load(hr_weights, map_location="cpu", weights_only=False)
    hr.load_state_dict(payload.get("model_state_dict", payload))

    model = network.Network(fe, hr).to(device).eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--threshold", type=float, default=5.0)
    ap.add_argument("--n-queries", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--weights", default=None, help="显式指定 hr 权重")
    ap.add_argument(
        "--zero-shot",
        action="store_true",
        help="用官方发布的权重(assets/dhe_vpr/official_dhe.torch)而非我们训的。"
        "method 名会变成 DHE-zeroshot,结果文件不与 target 版互相覆盖",
    )
    add_eval_args(ap)
    a = ap.parse_args()
    thresholds = parse_thresholds(a)

    from core.vpr_core import get_all_images_recursive
    import torchvision.transforms as transforms

    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    scenes = [s.strip() for s in a.scenes.split(",") if s.strip()]
    if len(scenes) != 1:
        raise SystemExit("--scenes 一次只能给一个场景:权重是逐场景训练的")

    if a.zero_shot:
        weights = a.weights or os.path.join(
            ROOT, "assets", "dhe_vpr", "official_dhe.torch"
        )
        if not os.path.isfile(weights):
            raise SystemExit(f"缺官方权重 {weights} —— 见 COMPAT_NOTES.md 的下载命令")
    else:
        weights = a.weights or find_weights(scenes[0], a.threshold)
    model = build_model(device, weights)
    print(f"[dhe] loaded {os.path.relpath(weights, ROOT)} on {device}")

    sys.path.insert(0, DHE)
    import homography_project

    transform = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    def extract(path):

        img = transform(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        with torch.inference_mode():
            feat = model.features_extractor(img)
        return feat[0].cpu()

    def score_candidates(query_feature, candidate_features):
        scores = []
        with torch.inference_mode():
            for cand in candidate_features:
                q = query_feature.unsqueeze(0).to(device)
                c = cand.unsqueeze(0).to(device)
                s = homography_project.compute_score(model, q, c)
                scores.append(float(s.reshape(-1)[0]))
        return scores

    evaluate_scenes(
        ROOT,
        a,
        ("DHE-zeroshot" if a.zero_shot else f"DHE-target-th{a.threshold:g}"),
        thresholds,
        [weights],
        get_all_images_recursive,
        extract,
        score_candidates,
        {
            "target_trained": not a.zero_shot,
            "train_threshold_m": a.threshold,
            "img_size": IMG_SIZE,
        },
    )


if __name__ == "__main__":
    main()
