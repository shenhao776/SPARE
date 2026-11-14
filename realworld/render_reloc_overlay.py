#!/usr/bin/env python


import argparse
import os

import numpy as np


def read_pcd(path, want=("x", "y", "z")):

    with open(path, "rb") as f:
        fields = sizes = types = counts = None
        n = 0
        data = "binary"
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: 头部没读到 DATA")
            k = line.decode("ascii", "replace").strip().split()
            if not k:
                continue
            if k[0] == "FIELDS":
                fields = k[1:]
            elif k[0] == "SIZE":
                sizes = [int(x) for x in k[1:]]
            elif k[0] == "TYPE":
                types = k[1:]
            elif k[0] == "COUNT":
                counts = [int(x) for x in k[1:]]
            elif k[0] == "POINTS":
                n = int(k[1])
            elif k[0] == "DATA":
                data = k[1]
                break
        if data != "binary":
            arr = np.loadtxt(path)
            return arr[:, [fields.index(w) for w in want]].astype(np.float32)
        if counts is None:
            counts = [1] * len(fields)
        dt = []
        for fname, sz, tp, ct in zip(fields, sizes, types, counts):
            base = {"F": "f", "U": "u", "I": "i"}[tp] + str(sz)
            dt.append((fname, base, ct) if ct > 1 else (fname, base))
        rec = np.frombuffer(
            f.read(n * sum(s * c for s, c in zip(sizes, counts))),
            dtype=np.dtype(dt),
            count=n,
        )
    return np.stack([rec[w].astype(np.float32) for w in want], axis=1)


def quat_to_R(qx, qy, qz, qw):
    n = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array(
        [
            [
                1 - 2 * (qy * qy + qz * qz),
                2 * (qx * qy - qz * qw),
                2 * (qx * qz + qy * qw),
            ],
            [
                2 * (qx * qy + qz * qw),
                1 - 2 * (qx * qx + qz * qz),
                2 * (qy * qz - qx * qw),
            ],
            [
                2 * (qx * qz - qy * qw),
                2 * (qy * qz + qx * qw),
                1 - 2 * (qx * qx + qy * qy),
            ],
        ]
    )


def nice_bar(span_m):

    target = span_m / 4.0
    for v in (1, 2, 5, 10, 20, 25, 50, 100, 200):
        if v >= target:
            return v
    return 200


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--map",
        default=os.path.expanduser(
            "~/datasets/my_map_data/PCD/map_final_visualization.pcd"
        ),
    )
    ap.add_argument("--scan", required=True)
    ap.add_argument(
        "--pose", required=True, nargs=7, type=float, help="x y z qx qy qz qw —— NDT 解"
    )
    ap.add_argument("--gt", nargs=2, type=float, default=None, help="真值 x y(绿)")
    ap.add_argument(
        "--center",
        nargs=2,
        type=float,
        default=None,
        help="画面中心。成/败两格必须给同一个值,否则两张图不可比",
    )
    ap.add_argument("--radius", type=float, default=15.0, help="纵向半宽(m)")
    ap.add_argument(
        "--slice-z",
        type=float,
        default=None,
        help="地图水平切片的参考高度。**成/败两格必须给同一个值** —— 默认跟着各自"
        "解出来的 z 走,而失败那次的 z 可能差好几米,于是两张底图的灰点根本"
        "不是同一批,肉眼一比就是'地图变了',但没有任何报错",
    )
    ap.add_argument(
        "--aspect",
        type=float,
        default=4.0 / 3.0,
        help="输出宽高比,要与排版里那一格的图区一致 —— 不一致的话排版会中央裁剪",
    )
    ap.add_argument(
        "--rotate-deg",
        type=float,
        default=0.0,
        help="整幅画面绕取景中心逆时针旋转的角度(地图、扫描、真值、箭头一起转)。"
        "用来把真值从角标底下挪开;成/败两格必须给同一个值",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--dpi", type=int, default=300)
    a = ap.parse_args()

    x, y, z, qx, qy, qz, qw = a.pose
    R = quat_to_R(qx, qy, qz, qw)
    cx, cy = a.center if a.center else (x, y)
    half_x, half_y = a.radius * a.aspect, a.radius

    m = read_pcd(a.map)
    hx, hy = (max(half_x, half_y),) * 2 if a.rotate_deg else (half_x, half_y)
    sel = (np.abs(m[:, 0] - cx) < hx) & (np.abs(m[:, 1] - cy) < hy)
    m = m[sel]
    zs = a.slice_z if a.slice_z is not None else z
    m = m[(m[:, 2] > zs - 1.0) & (m[:, 2] < zs + 4.5)]
    if len(m) > 500_000:
        m = m[np.random.default_rng(0).choice(len(m), 500_000, replace=False)]

    s = read_pcd(a.scan)
    r = np.linalg.norm(s[:, :2], axis=1)
    s = s[(r > 1.0) & (r < 60.0)]
    s = s @ R.T + np.array([x, y, z])
    if a.rotate_deg:
        th = np.deg2rad(a.rotate_deg)
        c_, s_ = np.cos(th), np.sin(th)
        rot2 = np.array([[c_, -s_], [s_, c_]])
        m[:, :2] = (m[:, :2] - [cx, cy]) @ rot2.T + [cx, cy]
        s[:, :2] = (s[:, :2] - [cx, cy]) @ rot2.T + [cx, cy]
        x, y = (rot2 @ (np.array([x, y]) - [cx, cy])) + [cx, cy]
        if a.gt:
            a.gt = list((rot2 @ (np.array(a.gt) - [cx, cy])) + [cx, cy])
        R3 = np.eye(3)
        R3[:2, :2] = rot2
        R = R3 @ R

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if a.out.lower().endswith(".pdf"):
        from matplotlib import font_manager

        arimo = "/usr/share/fonts/truetype/croscore/Arimo-Regular.ttf"
        if os.path.exists(arimo):
            font_manager.fontManager.addfont(arimo)
            matplotlib.rcParams["font.family"] = "Arimo"
        matplotlib.rcParams["pdf.fonttype"] = 42

    fig, ax = plt.subplots(figsize=(3.0 * a.aspect, 3.0))
    ax.scatter(m[:, 0], m[:, 1], s=2.4, c="#AFB6C2", linewidths=0, rasterized=True)
    ax.scatter(
        s[:, 0], s[:, 1], s=2.0, c="#D64545", linewidths=0, alpha=0.82, rasterized=True
    )
    if a.gt:
        ax.plot(
            [a.gt[0]],
            [a.gt[1]],
            marker="o",
            ms=17,
            mfc="none",
            mew=4.0,
            color="#2E7D32",
            zorder=5,
        )
        ax.plot(
            [a.gt[0]], [a.gt[1]], marker="+", ms=13, mew=2.6, color="#2E7D32", zorder=5
        )
    fwd = R @ np.array([1.0, 0.0, 0.0])
    L = 0.22 * a.radius
    ax.arrow(
        x,
        y,
        fwd[0] * L,
        fwd[1] * L,
        width=0.030 * a.radius,
        head_width=0.115 * a.radius,
        length_includes_head=True,
        color="#12489E",
        zorder=6,
    )
    ax.set_xlim(cx - half_x, cx + half_x)
    ax.set_ylim(cy - half_y, cy + half_y)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    bar = nice_bar(2 * half_x)
    bx0 = cx - half_x + 0.055 * 2 * half_x
    by0 = cy - half_y + 0.085 * 2 * half_y
    ax.plot(
        [bx0, bx0 + bar],
        [by0, by0],
        lw=3.2,
        color="#101318",
        zorder=7,
        solid_capstyle="butt",
    )
    ax.text(
        bx0 + bar / 2,
        by0 + 0.03 * 2 * half_y,
        f"{bar:g} m",
        ha="center",
        va="bottom",
        fontsize=18,
        color="#101318",
        zorder=7,
    )
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(a.out, dpi=a.dpi, facecolor="white")
    print(f"wrote {a.out}  map {len(m)} pts, scan {len(s)} pts, bar {bar:g} m")


if __name__ == "__main__":
    main()
