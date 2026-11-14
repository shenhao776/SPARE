#!/usr/bin/env python


import argparse
import json
import os
import runpy
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SELA = os.path.join(ROOT, "3rdparty", "SelaVPR")
MAPS = [
    "RobotCar",
    "BusinessCampus",
    "Neighborhood",
    "OldTown",
    "NCLT",
    "Boreas",
    "CustomCampus",
]
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def val_r1(model, local_sim, device, limit=None):

    import torch
    import torchvision.transforms as T
    from PIL import Image

    tf = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])

    def load(path):
        image = Image.open(os.path.join(ROOT, path)).convert("RGB")
        return tf(T.functional.resize(image, [224, 224]))

    per_map = {}
    for scene in MAPS:
        places = os.path.join(ROOT, "assets", "pairvpr_target", f"places_{scene}.json")
        with open(places, encoding="utf-8") as handle:
            val = json.load(handle)["val"][:limit]
        hit = 0
        for v in val:
            batch = torch.stack([load(p) for p in [v["query"]] + v["cands"]])
            with torch.inference_mode():
                local, _ = model(batch.to(device))
                scores = local_sim(local[:1], local[1:]).cpu().tolist()
            best = max(range(len(scores)), key=lambda i: (scores[i], -i))
            hit += v["err"][best] <= 5.0
        per_map[scene] = round(100.0 * hit / max(1, len(val)), 1)
    return sum(per_map.values()) / len(per_map), per_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "assets", "ft_data"))
    ap.add_argument("--name", default="UnionMaps")
    ap.add_argument("--save-dir", default="union_ft")
    ap.add_argument("--radius", type=int, default=5)
    ap.add_argument("--queries-per-epoch", type=int, default=5000)
    ap.add_argument("--cache-refresh-rate", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=50, help="official default")
    ap.add_argument("--batch", type=int, default=4, help="official default")
    ap.add_argument("--negs", type=int, default=2, help="official default")
    ap.add_argument("--workers", type=int, default=8, help="official default")
    ap.add_argument("--val-limit", type=int, default=None, help="smoke tests only")
    ap.add_argument(
        "--vram-frac",
        type=float,
        default=None,
        help="cap this process's GPU memory (fraction of the card), e.g. 0.75 "
        "on a machine whose desktop shares the GPU",
    )
    ap.add_argument(
        "--weights", default=os.path.join(SELA, "weights", "SelaVPR_msls.pth")
    )
    ap.add_argument(
        "--continue-run",
        default=None,
        help="folder of an interrupted run (logs/<save-dir>/<time>): continue it "
        "from its last_model.pth",
    )
    a = ap.parse_args()
    if a.continue_run:
        a.continue_run = os.path.abspath(a.continue_run)

    import numpy as np

    np.float = float
    np.int = int

    official_delete = np.delete

    def delete_ragged(arr, obj, axis=None):
        if isinstance(arr, list) and arr and isinstance(arr[0], np.ndarray):
            boxed = np.empty(len(arr), dtype=object)
            for i, item in enumerate(arr):
                boxed[i] = item
            arr = boxed
        return official_delete(arr, obj, axis)

    np.delete = delete_ragged
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    import logging

    logging.getLogger("PIL").setLevel(logging.WARNING)
    sys.path.insert(0, SELA)
    os.chdir(SELA)

    import shutil

    import torch

    torch.multiprocessing.set_sharing_strategy("file_system")
    if a.vram_frac:
        torch.cuda.set_per_process_memory_fraction(a.vram_frac)
    held = {}
    seed = 0
    if a.continue_run:
        held["ckpt"] = torch.load(
            os.path.join(a.continue_run, "last_model.pth"),
            map_location="cpu",
            weights_only=False,
        )
        seed = held["ckpt"]["epoch_num"] + 1
    official_adam = torch.optim.Adam

    class RecordedAdam(official_adam):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            held["optimizer"] = self

    torch.optim.Adam = RecordedAdam
    import util
    import test as sela_test
    from local_matching import local_sim

    def resume_model_only(args, model, optimizer=None, strict=False):
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        print(f"[selavpr-target] model <- {args.resume} (training state reset)")
        net = model.module if hasattr(model, "module") else model
        net.eval()
        v0, m0 = val_r1(net, local_sim, args.device, a.val_limit)
        print(f"[selavpr-target] epoch 0 (released weights) val R@1@5m {v0:.2f} {m0}")
        return model, optimizer, 0, 0, 0

    def resume_continue(args, model, optimizer=None, strict=False):
        ckpt = held.pop("ckpt")
        model.load_state_dict(ckpt["model_state_dict"], strict=strict)
        held["optimizer"].load_state_dict(ckpt["optimizer_state_dict"])
        last = float(ckpt["recalls"][1])
        if last > ckpt["best_r5"]:
            best, waited = last, 0
        else:
            best, waited = ckpt["best_r5"], ckpt["not_improved_num"] + 1
        shutil.copy(os.path.join(a.continue_run, "best_model.pth"), args.save_dir)
        start = ckpt["epoch_num"] + 1
        print(
            f"[selavpr-target] continuing {a.continue_run}: epoch {start}, "
            f"best val {best:.2f}, {waited} epoch(s) without improvement, "
            f"optimizer state restored, seed {args.seed}",
            flush=True,
        )
        return model, optimizer, best, start, waited

    def validate(args, eval_ds, model, test_method="hard_resize", pca=None):
        net = model.module if hasattr(model, "module") else model
        was_training = net.training
        net.eval()
        value, per_map = val_r1(net, local_sim, args.device, a.val_limit)
        if was_training:
            net.train()
        text = f"val R@1@5m {value:.2f} {per_map}"
        print(f"[selavpr-target] {text}", flush=True)
        return np.full(len(args.recall_values), value), text

    import datasets_ws

    class DiscardedLocalCache:

        def __init__(self, shape, dtype=None):
            self.shape = shape

        def __setitem__(self, indexes, vals):
            pass

    datasets_ws.RAMEfficient4DMatrix = DiscardedLocalCache

    if a.continue_run:
        util.resume_train = resume_continue
        resume_from = os.path.join(a.continue_run, "last_model.pth")
    else:
        util.resume_train = resume_model_only
        resume_from = a.weights
    sela_test.test = validate

    sys.argv = [
        "train.py",
        "--datasets_folder",
        a.data,
        "--dataset_name",
        a.name,
        "--resume",
        resume_from,
        "--seed",
        str(seed),
        "--train_positives_dist_threshold",
        str(a.radius),
        "--val_positive_dist_threshold",
        str(a.radius),
        "--queries_per_epoch",
        str(a.queries_per_epoch),
        "--cache_refresh_rate",
        str(a.cache_refresh_rate),
        "--epochs_num",
        str(a.epochs),
        "--train_batch_size",
        str(a.batch),
        "--negs_num_per_query",
        str(a.negs),
        "--num_workers",
        str(a.workers),
        "--save_dir",
        a.save_dir,
    ]
    runpy.run_path(os.path.join(SELA, "train.py"), run_name="__main__")


if __name__ == "__main__":
    main()
