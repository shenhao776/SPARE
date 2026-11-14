#!/usr/bin/env python


import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

_PLACEHOLDERS = "@" * 11


def _encode_name(x, y, original):

    stem = os.path.splitext(os.path.basename(original))[0]
    return f"@{x:.3f}@{y:.3f}{_PLACEHOLDERS}@{stem}.jpg"


def _link_group(paths, pose_manager, dst_dir, label):

    os.makedirs(dst_dir, exist_ok=True)
    linked = skipped = 0
    for p in paths:
        info = pose_manager.get_pose_info(os.path.abspath(p))
        if not info or "global" not in info:
            skipped += 1
            continue
        xy = info["global"][:2]
        dst = os.path.join(dst_dir, _encode_name(xy[0], xy[1], p))
        if os.path.islink(dst) or os.path.exists(dst):
            linked += 1
            continue
        os.symlink(os.path.abspath(p), dst)
        linked += 1
    print(
        f"    {label}: 链接 {linked} 张"
        + (f",跳过无位姿 {skipped} 张" if skipped else "")
    )
    return linked


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", required=True, help="场景名,与 datasets/<scene> 同名")
    ap.add_argument(
        "--out",
        default=os.path.join(ROOT, "assets", "prgs_data"),
        help="导出根目录。默认放 assets/ 下 —— 它是可重建层,删了再跑一次即可",
    )
    ap.add_argument(
        "--splits",
        default="train,val,test",
        help="要导出的 split(逗号分隔)。只训练时 test 也要导,PRGS 每个 split 都要求存在",
    )
    a = ap.parse_args()

    from core.vpr_core import get_all_images_recursive, pose_manager

    scene_src = os.path.join(ROOT, "datasets", a.scene)
    if not os.path.isdir(scene_src):
        raise SystemExit(
            f"找不到 {scene_src}\n"
            f"  「有没有这个场景」看 datasets/ 这一层,不看 assets/。"
        )

    db_src = os.path.join(scene_src, "database")
    if not os.path.isdir(db_src):
        raise SystemExit(f"找不到 {db_src} —— 建图库缺失,无法导出")
    db_paths = sorted(get_all_images_recursive(db_src))

    print(f"[{a.scene}] 导出到 {a.out}")
    total = 0
    for split in [s.strip() for s in a.splits.split(",") if s.strip()]:
        q_src = os.path.join(scene_src, f"queries_{split}")
        if not os.path.isdir(q_src):
            print(f"  [{split}] 跳过:没有 {os.path.basename(q_src)}")
            continue
        q_paths = sorted(get_all_images_recursive(q_src))
        base = os.path.join(a.out, a.scene, "images", split)
        print(f"  [{split}]")
        total += _link_group(
            db_paths, pose_manager, os.path.join(base, "database"), "database"
        )
        total += _link_group(
            q_paths, pose_manager, os.path.join(base, "queries"), "queries"
        )

        gallery = os.path.join(base, "gallery")
        if not os.path.lexists(gallery):
            os.symlink("database", gallery)

    if total == 0:
        raise SystemExit("一张都没导出 —— 检查场景名与 queries_* 目录是否存在")
    print(f"[{a.scene}] 完成,共 {total} 个链接")
    print(
        "\n接下来用**官方**入口训练(不经我们的代码):\n"
        f"  cd 3rdparty/PRGS && python train_reranker.py \\\n"
        f"      --datasets_folder {a.out} --dataset_name {a.scene} \\\n"
        f"      --train_positives_dist_threshold <评测半径>"
    )


if __name__ == "__main__":
    main()
