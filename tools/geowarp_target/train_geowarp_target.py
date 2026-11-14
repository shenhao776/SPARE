#!/usr/bin/env python


import argparse
import glob
import os
import runpy
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GEO_WARP = os.path.join(ROOT, "3rdparty", "geo_warp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "assets", "ft_data"))
    ap.add_argument("--name", default="UnionMaps")
    ap.add_argument("--exp-name", default="union_ft")
    ap.add_argument("--radius", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=100, help="official default")
    ap.add_argument("--iters", type=int, default=500, help="official default")
    ap.add_argument(
        "--mining-cache",
        default=None,
        help="path prefix: store / reuse the query-positive mining result (the "
        "extractor is frozen, so it is the same for every run on the same data)",
    )
    ap.add_argument(
        "--duty-file",
        default="/tmp/spare_geowarp_duty",
        help="file holding the fraction of wall time to compute (e.g. 0.4); "
        "read every step, absent = full speed",
    )
    ap.add_argument(
        "--fe-chunk",
        type=int,
        default=4,
        help="images per gradient-checkpointed chunk of the frozen extractor "
        "(memory only; gradients are identical for any value)",
    )
    ap.add_argument(
        "--from-scratch",
        action="store_true",
        help="do not load the released homography regression",
    )
    ap.add_argument(
        "--continue-run",
        default=None,
        help="folder of a run (runs/<exp-name>/<time>): continue it after its last "
        "saved epoch",
    )
    a = ap.parse_args()
    a.data = os.path.abspath(a.data)
    if a.mining_cache:
        a.mining_cache = os.path.abspath(a.mining_cache)
    start = 0
    if a.continue_run:
        a.continue_run = os.path.abspath(a.continue_run)
        saved = sorted(
            glob.glob(
                os.path.join(
                    a.continue_run, "checkpoints", "homography_regression_*.torch"
                )
            )
        )
        if not saved:
            raise SystemExit(f"no checkpoint in {a.continue_run}")
        start = int(os.path.basename(saved[-1]).split("_")[-1].split(".")[0]) + 1
        if start >= a.epochs:
            raise SystemExit(f"{a.continue_run} already has {start} epochs")

    import numpy as np

    np.float = float
    np.int = int

    fe_ckpt = os.path.join(
        GEO_WARP, "data", "pretrained_baselines", "resnet50_netvlad.pth"
    )
    hr_ckpt = os.path.join(
        GEO_WARP, "data", "trained_homography_regressions", "resnet50_netvlad.pth"
    )
    sys.path.insert(0, GEO_WARP)
    os.chdir(GEO_WARP)

    import datasets_util
    from PIL import Image

    def open_rgb_640x480(image_path):
        image = Image.open(image_path).convert("RGB").resize((640, 480), Image.BILINEAR)
        return datasets_util.transform(image)

    datasets_util.open_image_and_apply_transform = open_rgb_640x480

    import torch
    import util
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    class TrainingDone(Exception):
        pass

    def lean_compute_features(
        geoloc_dataset,
        model,
        global_features_dim,
        num_workers=4,
        eval_batch_size=32,
        recall_values=[1, 5, 10, 20],
    ):

        if os.path.basename(geoloc_dataset.dataset_folder.rstrip("/")) == "test":
            raise TrainingDone
        cache = None
        if a.mining_cache:
            cache = f"{a.mining_cache}_{len(geoloc_dataset)}_{geoloc_dataset.gallery_num}.npz"
        if cache and os.path.isfile(cache):
            z = np.load(cache)
            print(f"[geowarp-target] mining result <- {cache}", flush=True)
            return (
                z["recalls"],
                str(z["recalls_str"]),
                z["predictions"],
                z["correct"],
                z["distances"],
            )
        loader = DataLoader(
            dataset=geoloc_dataset,
            num_workers=num_workers,
            batch_size=eval_batch_size,
            pin_memory=True,
        )
        model = model.eval()
        feats = np.empty((len(geoloc_dataset), global_features_dim), dtype=np.float16)
        with torch.no_grad():
            for inputs, indices in tqdm(loader, desc="Comp feats", ncols=100):
                out = model("features_extractor", [inputs.cuda(), "global"])
                out = out.reshape(inputs.shape[0], global_features_dim)
                feats[indices.numpy(), :] = out.half().cpu().numpy()
            n_gal = geoloc_dataset.gallery_num
            gallery = torch.from_numpy(feats[:n_gal]).cuda()
            k = max(recall_values)
            g_step, q_step = 4096, 256
            g_sq = torch.cat(
                [
                    (gallery[s : s + g_step].float() ** 2).sum(1)
                    for s in range(0, n_gal, g_step)
                ]
            )
            distances, predictions = [], []
            for qs in range(n_gal, len(feats), q_step):
                q = torch.from_numpy(feats[qs : qs + q_step]).cuda().float()
                dots = torch.cat(
                    [
                        q @ gallery[s : s + g_step].float().T
                        for s in range(0, n_gal, g_step)
                    ],
                    dim=1,
                )
                d = (q**2).sum(1, keepdim=True) + g_sq[None, :] - 2.0 * dots
                dist, idx = torch.topk(d, k, dim=1, largest=False)
                distances.append(dist.cpu().numpy())
                predictions.append(idx.cpu().numpy())
            del gallery, g_sq, q, dots, d, dist, idx
            torch.cuda.empty_cache()
        distances = np.concatenate(distances).astype(np.float32)
        predictions = np.concatenate(predictions)
        ground_truths = geoloc_dataset.get_positives()
        recalls, recalls_str = util.compute_recalls(
            predictions, ground_truths, geoloc_dataset, recall_values
        )
        correct = np.zeros(predictions.shape, dtype=int)
        for query_index in range(geoloc_dataset.queries_num):
            positives = set(ground_truths[query_index].tolist())
            for pred_index in range(k):
                if predictions[query_index, pred_index] in positives:
                    correct[query_index, pred_index] = 1
        if cache:
            np.savez(
                cache,
                recalls=recalls,
                recalls_str=recalls_str,
                predictions=predictions,
                correct=correct,
                distances=distances,
            )
        return recalls, recalls_str, predictions, correct, distances

    util.compute_features = lean_compute_features

    import dataset_qp

    official_qp = dataset_qp.DatasetQP

    class CountedQP(official_qp):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            print(f"[geowarp-target] {len(self)} query-positive pairs", flush=True)

    dataset_qp.DatasetQP = CountedQP

    import time

    official_sgd = torch.optim.SGD

    class DutyCycledSGD(official_sgd):

        _mark = None

        def step(self, *args, **kwargs):
            out = super().step(*args, **kwargs)
            try:
                with open(a.duty_file, encoding="utf-8") as handle:
                    duty = float(handle.read().strip() or 1.0)
            except (OSError, ValueError):
                duty = 1.0
            if 0.0 < duty < 1.0:
                torch.cuda.synchronize()
                if self._mark is not None:
                    time.sleep((time.monotonic() - self._mark) * (1.0 / duty - 1.0))
            self._mark = time.monotonic()
            return out

    torch.optim.SGD = DutyCycledSGD

    import network
    from torch.utils.checkpoint import checkpoint

    official_fe_forward = network.FeaturesExtractor.forward

    def lean_fe_forward(self, x, f_type="local"):

        if next(self.parameters()).requires_grad:
            for param in self.parameters():
                param.requires_grad_(False)
        if torch.is_grad_enabled() and x.requires_grad and x.shape[0] > a.fe_chunk:
            parts = [
                checkpoint(official_fe_forward, self, c, f_type, use_reentrant=False)
                for c in x.split(a.fe_chunk)
            ]
            return torch.cat(parts)
        return official_fe_forward(self, x, f_type)

    network.FeaturesExtractor.forward = lean_fe_forward

    init_ckpt = saved[-1] if a.continue_run else None
    if init_ckpt is None and not a.from_scratch:
        init_ckpt = hr_ckpt
    if init_ckpt:
        init_state = torch.load(init_ckpt, map_location="cpu")
        official = network.HomographyRegression

        class InitFromCheckpoint(official):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.load_state_dict(init_state)
                print(f"[geowarp-target] homography regression <- {init_ckpt}")

        network.HomographyRegression = InitFromCheckpoint

    if a.continue_run:
        official_save = torch.save

        def save_continued(obj, f, *args, **kwargs):

            name = os.path.basename(str(f))
            if name.startswith("homography_regression_") and name.endswith(".torch"):
                k = start + int(name.split("_")[-1].split(".")[0])
                f = os.path.join(
                    a.continue_run,
                    "checkpoints",
                    f"homography_regression_{k:03d}.torch",
                )
            return official_save(obj, f, *args, **kwargs)

        torch.save = save_continued
        print(
            f"[geowarp-target] continuing {a.continue_run} at epoch index {start} "
            f"({a.epochs - start} epochs left; SGD momentum restarts from zero)",
            flush=True,
        )

    sys.argv = [
        "train.py",
        "--datasets_folder",
        a.data,
        "--dataset_name",
        a.name,
        "--positive_dist_threshold",
        str(a.radius),
        "--resume_fe",
        fe_ckpt,
        "--exp_name",
        a.exp_name,
        "--n_epochs",
        str(a.epochs - start),
        "--iterations_per_epoch",
        str(a.iters),
    ]
    try:
        runpy.run_path(os.path.join(GEO_WARP, "train.py"), run_name="__main__")
    except TrainingDone:
        print("[geowarp-target] training finished; final test on the val link skipped")


if __name__ == "__main__":
    main()
