import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "3rdparty", "Pair-VPR"))
os.chdir(ROOT)

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


def _sdpa_attention():

    import torch.nn.functional as F
    from pairvpr.models.tools import blocks

    def attn_forward(self, x, xpos=None, attn_bias=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        drop = self.attn_drop.p if self.training else 0.0
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=drop)
        return self.proj_drop(self.proj(x.transpose(1, 2).reshape(B, N, C)))

    def cross_forward(self, query, key, value, qpos=None, kpos=None, attn_bias=None):
        B, Nq, C = query.shape
        h = self.num_heads
        q = self.projq(query).reshape(B, Nq, h, C // h).transpose(1, 2)
        k = self.projk(key).reshape(B, key.shape[1], h, C // h).transpose(1, 2)
        v = self.projv(value).reshape(B, value.shape[1], h, C // h).transpose(1, 2)
        drop = self.attn_drop.p if self.training else 0.0
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=drop)
        return self.proj_drop(self.proj(x.transpose(1, 2).reshape(B, Nq, C)))

    for cls in (blocks.Attention, blocks.MemEffAttention):
        cls.forward = attn_forward
    for cls in (blocks.CrossAttention, blocks.MemEffCrossAttention):
        cls.forward = cross_forward

    import torch.utils.checkpoint as cp

    orig = blocks.DecoderBlock.forward

    def ckpt_forward(self, x, y, xpos=None, ypos=None):
        if self.training and torch.is_grad_enabled():
            return cp.checkpoint(orig, self, x, y, xpos, ypos, use_reentrant=False)
        return orig(self, x, y, xpos, ypos)

    blocks.DecoderBlock.forward = ckpt_forward


def _offline_hub():

    hub_dir = torch.hub.get_dir()
    orig = torch.hub.load

    def load(repo_or_dir, model, *args, **kwargs):
        if kwargs.get("source", "github") == "github" and "/" in repo_or_dir:
            owner, name = repo_or_dir.split(":")[0].split("/")
            local = os.path.join(hub_dir, f"{owner}_{name}_main")
            if os.path.isdir(local):
                kwargs["source"] = "local"
                kwargs.pop("trust_repo", None)
                return orig(local, model, *args, **kwargs)
        return orig(repo_or_dir, model, *args, **kwargs)

    torch.hub.load = load


def build_model(device):
    from tools.run_pairvpr_s2_rerank import build_model as _build

    _offline_hub()

    model = _build(device)
    _sdpa_attention()
    return model


class Places(torch.utils.data.Dataset):

    def __init__(self, maps, radius, img_per_place=4):
        self.items = []
        self.xy = []
        for mi, scene in enumerate(maps):
            d = json.load(open(f"assets/pairvpr_target/places_{scene}.json"))
            db = d["database"]
            tree = cKDTree(np.array([[x, y] for _p, x, y in db]))
            for qp, x, y in d["queries"]:
                pos = [db[j][0] for j in tree.query_ball_point([x, y], radius)]
                if not pos:
                    continue
                self.items.append((mi, qp, pos))
                self.xy.append((x, y))
        self.n = img_per_place
        self.tf = T.Compose(
            [
                T.Resize((322, 322), interpolation=T.InterpolationMode.BILINEAR),
                T.RandAugment(num_ops=3, interpolation=T.InterpolationMode.BILINEAR),
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD),
            ]
        )
        self.epoch_seed = 0

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        _mi, qp, pos = self.items[i]
        rng = random.Random(f"{self.epoch_seed}/{i}")
        k = self.n - 1
        picks = (
            rng.sample(pos, k) if len(pos) >= k else [rng.choice(pos) for _ in range(k)]
        )
        ims = [self.tf(Image.open(p).convert("RGB")) for p in [qp] + picks]
        return torch.stack(ims), torch.full((self.n,), i, dtype=torch.long)


class PlaceBatches(torch.utils.data.Sampler):

    def __init__(self, ds, bs, min_sep, seed):
        self.ds, self.bs, self.min_sep, self.seed = ds, bs, min_sep, seed
        self.epoch = 0

    def __iter__(self):
        rng = random.Random(f"{self.seed}/{self.epoch}")
        by_map = {}
        for i, (mi, _q, _p) in enumerate(self.ds.items):
            by_map.setdefault(mi, []).append(i)
        batches = []
        for idx in by_map.values():
            rng.shuffle(idx)
            pending = list(idx)
            while len(pending) >= self.bs:
                batch, rest = [], []
                for i in pending:
                    xi = self.ds.xy[i]
                    if len(batch) < self.bs and all(
                        math.dist(xi, self.ds.xy[j]) >= self.min_sep for j in batch
                    ):
                        batch.append(i)
                    else:
                        rest.append(i)
                if len(batch) < self.bs:
                    break
                batches.append(batch)
                pending = rest
        rng.shuffle(batches)
        return iter(batches)

    def __len__(self):
        return sum(1 for _ in self.__iter__())


@torch.no_grad()
def validate(model, device, maps, limit=None):

    tf = T.Compose(
        [T.ToTensor(), T.Resize((322, 322), antialias=True), T.Normalize(MEAN, STD)]
    )
    model.eval()
    per_map = {}
    for scene in maps:
        val = json.load(open(f"assets/pairvpr_target/places_{scene}.json"))["val"][
            :limit
        ]
        hit = 0
        for v in val:
            ims = torch.stack(
                [tf(Image.open(p).convert("RGB")) for p in [v["query"]] + v["cands"]]
            )
            with torch.autocast("cuda"):
                feats, _ = model(ims.to(device), None, mode="global")
                q = feats[:1].expand(len(v["cands"]), *feats.shape[1:])
                d = feats[1:]
                s = (model(q, d, "pairvpr") + model(d, q, "pairvpr")).flatten().float()
            hit += v["err"][int(s.argmax())] <= 5.0
        per_map[scene] = 100.0 * hit / max(1, len(val))
    model.train()
    return sum(per_map.values()) / len(per_map), per_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument(
        "--places", type=int, default=8, help="places per batch (official 100)"
    )
    ap.add_argument("--radius", type=float, default=5.0)
    ap.add_argument("--min-sep", type=float, default=25.0)
    ap.add_argument(
        "--max-iters", type=int, default=0, help="smoke test: stop after N iterations"
    )
    ap.add_argument("--val-limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--mem-frac",
        type=float,
        default=0.8,
        help="per-process CUDA memory cap (fraction of the card); jp needs room for Windows",
    )
    a = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(a.mem_frac)

    from pairvpr.training.losses import VPRLoss

    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    device = torch.device("cuda")
    model = build_model(device)
    model.train()
    os.makedirs(a.out, exist_ok=True)
    log = open(os.path.join(a.out, "train_log.jsonl"), "a")

    ds = Places(MAPS, a.radius)
    sampler = PlaceBatches(ds, a.places, a.min_sep, seed=0)
    loader = torch.utils.data.DataLoader(
        ds, batch_sampler=sampler, num_workers=a.workers, pin_memory=True
    )
    iters_per_epoch = len(sampler)
    total = iters_per_epoch * a.epochs
    lr = 8e-5 * a.places / 100.0
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.05
    )
    sched = torch.optim.lr_scheduler.LinearLR(opt, 1.0, 0.2, total_iters=total)
    scaler = torch.amp.GradScaler("cuda")
    lossobj = VPRLoss()
    bce = torch.nn.BCEWithLogitsLoss()
    print(
        f"[pairvpr-target] {len(ds)} places, {iters_per_epoch} it/epoch x {a.epochs}, "
        f"lr {lr:.2e}, trainable params "
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.1f} M",
        flush=True,
    )

    v0, m0 = validate(model, device, MAPS, a.val_limit)
    print(
        f"[pairvpr-target] epoch 0 (released weights) val R@1@5m {v0:.2f} {m0}",
        flush=True,
    )
    log.write(json.dumps({"epoch": 0, "val": v0, "per_map": m0}) + "\n")
    log.flush()
    best = (v0, 0)
    it = 0
    for epoch in range(1, a.epochs + 1):
        sampler.epoch = epoch
        ds.epoch_seed = epoch
        t0 = time.time()
        for places, labels in loader:
            BS, N, ch, h, w = places.shape
            images = places.to(device, non_blocking=True).view(BS * N, ch, h, w)
            labels = labels.view(-1)
            with torch.amp.autocast("cuda"):
                feats, desc = model(images, None, "global")
                mined_ms, mined_bh = lossobj.mining(desc, labels)
                loss1, _acc = lossobj.loss_function_global(desc, labels, mined_ms)
                dp = model(feats[mined_bh[0]], feats[mined_bh[1]], "pairvpr").flatten()
                dn = model(feats[mined_bh[0]], feats[mined_bh[2]], "pairvpr").flatten()
                tgt = torch.cat((torch.ones_like(dp), torch.zeros_like(dn)))
                loss2 = bce(torch.cat((dp, dn)).float(), tgt.float())
                loss = loss1 + 2.0 * loss2
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            opt.zero_grad(set_to_none=True)
            it += 1
            if it % 50 == 0 or it <= 3:
                mem = torch.cuda.max_memory_allocated() / 2**30
                print(
                    f"  ep {epoch} it {it}/{total} loss {loss.item():.4f} "
                    f"(global {loss1.item():.4f}, pair {loss2.item():.4f}) "
                    f"{(time.time() - t0) / max(1, it - (epoch - 1) * iters_per_epoch):.2f}s/it "
                    f"peak {mem:.1f} GB",
                    flush=True,
                )
            if a.max_iters and it >= a.max_iters:
                break
        ck = os.path.join(a.out, f"epoch{epoch}.pth")
        torch.save(model.state_dict(), ck)
        v, m = validate(model, device, MAPS, a.val_limit)
        print(f"[pairvpr-target] epoch {epoch} val R@1@5m {v:.2f} {m}", flush=True)
        log.write(
            json.dumps({"epoch": epoch, "val": v, "per_map": m, "ckpt": ck}) + "\n"
        )
        log.flush()
        if v > best[0]:
            best = (v, epoch)
        if a.max_iters and it >= a.max_iters:
            break
    json.dump(
        {"best_epoch": best[1], "best_val": best[0], "released_val": v0},
        open(os.path.join(a.out, "selection.json"), "w"),
        indent=2,
    )
    print(
        f"[pairvpr-target] done: best epoch {best[1]} (val {best[0]:.2f}; released {v0:.2f})"
    )


if __name__ == "__main__":
    main()
