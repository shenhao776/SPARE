#!/usr/bin/env python


import argparse
import os
import sys

import torch
import torch.nn.functional as F
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
R2F = os.path.join(ROOT, "3rdparty", "R2Former")
sys.path.insert(0, R2F)
from tools.rerank_eval_common import (
    add_eval_args,
    evaluate_scenes,
    parse_thresholds,
)


def build_model(device):
    from argparse import Namespace

    import sys, types

    if "google_drive_downloader" not in sys.modules:
        stub = types.ModuleType("google_drive_downloader")

        class _GDD:
            @staticmethod
            def download_file_from_google_drive(*a, **k):
                raise RuntimeError("unexpected google-drive download at test time")

        stub.GoogleDriveDownloader = _GDD
        sys.modules["google_drive_downloader"] = stub
    from model.network import GeoLocalizationNetRerank as Net

    args = Namespace(
        backbone="deit",
        fc_output_dim=256,
        aggregation="gem",
        resize=[480, 640],
        local_dim=128,
        num_local=500,
        finetune=0,
        hypercolumn=0,
        rerank_model="r2former",
    )
    model = Net(args).to(device)
    ckpt = torch.load(
        os.path.join(R2F, "R2Former", "CVPR23_DeitS_Rerank.pth"),
        map_location="cpu",
        weights_only=False,
    )
    sd = ckpt["model_state_dict"]
    if any(k.startswith("module.") for k in sd):
        sd = {k[len("module.") :]: v for k, v in sd.items()}
    load_report = model.load_state_dict(sd, strict=False)
    if load_report.missing_keys:
        raise RuntimeError(
            "R2Former checkpoint left parameters uninitialized (a key drift "
            f"would silently mean random weights): {load_report.missing_keys[:8]}"
        )
    model.eval()
    return model, args


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--n-queries", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    add_eval_args(ap)
    a = ap.parse_args()
    thresholds = parse_thresholds(a)

    from core.vpr_core import get_all_images_recursive
    import torchvision.transforms as transforms

    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    model, model_args = build_model(device)
    ckpt_path = os.path.join(R2F, "R2Former", "CVPR23_DeitS_Rerank.pth")
    print(f"[r2former] loaded CVPR23_DeitS_Rerank on {device}")
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    def extract(path):
        image = (
            Image.open(path)
            .convert("RGB")
            .resize((model_args.resize[1], model_args.resize[0]), Image.BILINEAR)
        )
        tensor = transform(image).unsqueeze(0).to(device)
        with torch.inference_mode():
            global_feature, local_feature = model(tensor)
        return global_feature[0].cpu(), local_feature[0].cpu()

    def score_candidates(query_feature, candidate_features):
        query_global, query_local = query_feature
        scores = []
        with torch.inference_mode():
            for database_global, database_local in candidate_features:
                logits, _ = model.Reranker(
                    query_global.unsqueeze(0).to(device),
                    query_local.unsqueeze(0).to(device),
                    database_global.unsqueeze(0).to(device),
                    database_local.unsqueeze(0).to(device),
                )
                scores.append(F.softmax(logits, dim=1)[0, 1].item())
        return scores

    evaluate_scenes(
        ROOT,
        a,
        "R2Former-rerank",
        thresholds,
        [ckpt_path],
        get_all_images_recursive,
        extract,
        score_candidates,
        {
            "resize": list(model_args.resize),
        },
    )


if __name__ == "__main__":
    main()
