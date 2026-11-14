import argparse
import os
import re

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DHE_LOGS = os.path.join(ROOT, "3rdparty", "DHE-VPR", "logs_trainDHE")
ARCHIVE = os.path.join(ROOT, "assets", "dhe_vpr")


def newest_checkpoint(tag):

    exp = os.path.join(DHE_LOGS, f"star_{tag}")
    if not os.path.isdir(exp):
        return None
    for run in sorted(os.listdir(exp), reverse=True):
        ckpt_dir = os.path.join(exp, run, "checkpoints")
        if not os.path.isdir(ckpt_dir):
            continue
        cands = []
        for name in os.listdir(ckpt_dir):
            m = re.match(r"homography_regression_(\d+)\.torch$", name)
            if m:
                cands.append((int(m.group(1)), os.path.join(ckpt_dir, name)))
        if cands:
            return max(cands)[1]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="<场景>_th<阈值>[_smoke]")
    a = ap.parse_args()

    src = newest_checkpoint(a.tag)
    if src is None:
        raise SystemExit(f"没找到 star_{a.tag} 的 homography_regression_*.torch")
    os.makedirs(ARCHIVE, exist_ok=True)
    dst = os.path.join(ARCHIVE, f"{a.tag}_hr.pth")

    state = torch.load(src, map_location="cpu", weights_only=False)
    torch.save(
        {
            "model_state_dict": state,
            "source_run": os.path.relpath(src, DHE_LOGS),
            "epoch_num": int(re.search(r"_(\d+)\.torch$", src).group(1)),
        },
        dst,
    )
    print(
        f"[归档] {os.path.relpath(dst, ROOT)}  "
        f"{os.path.getsize(src) / 1048576:.0f}MB → {os.path.getsize(dst) / 1048576:.0f}MB"
    )


if __name__ == "__main__":
    main()
