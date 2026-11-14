#!/usr/bin/env python


import argparse
import os

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RELEASED = os.path.join(ROOT, "3rdparty", "SelaVPR", "weights", "SelaVPR_msls.pth")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("export", "apply"))
    ap.add_argument("--released", default=RELEASED)
    ap.add_argument("--tuned")
    ap.add_argument("--delta")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    base = torch.load(a.released, map_location="cpu", weights_only=False)[
        "model_state_dict"
    ]
    if a.mode == "export":
        tuned = torch.load(a.tuned, map_location="cpu", weights_only=False)
        state = tuned["model_state_dict"]
        if set(state) != set(base):
            raise SystemExit("fine-tuned and released checkpoints have different keys")
        delta = {k: v for k, v in state.items() if not torch.equal(v, base[k])}
        meta = {k: tuned[k] for k in ("epoch_num", "recalls") if k in tuned}
        torch.save({"delta": delta, "meta": meta}, a.out)
        size = os.path.getsize(a.out) / 1e6
        print(f"{len(delta)}/{len(state)} tensors differ -> {a.out} ({size:.1f} MB)")
    else:
        pack = torch.load(a.delta, map_location="cpu", weights_only=False)
        state = dict(base)
        state.update(pack["delta"])
        torch.save({"model_state_dict": state, **pack.get("meta", {})}, a.out)
        print(f"{len(pack['delta'])} tensors replaced -> {a.out}")


if __name__ == "__main__":
    main()
