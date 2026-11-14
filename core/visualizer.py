import math
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R_sci
from PIL import Image, ImageDraw


def generate_topk_pair(
    query_path, candidate, topk_dir, dataset_dir, query_pose_info, dst_name
):
    def load_img(path):
        if not path or not os.path.exists(path):
            return Image.new("RGB", (300, 300), color=(50, 50, 50))
        return Image.open(path).convert("RGB")

    basename = os.path.basename(query_path)
    queries_dir = os.path.join(dataset_dir, "queries_test")

    if query_path.startswith(queries_dir):
        rel_path = os.path.relpath(query_path, queries_dir)
        debug_rel_dir = os.path.dirname(rel_path)
        debug_img_path = os.path.join(
            dataset_dir,
            "queries_debug_test",
            debug_rel_dir,
            f"debug_{os.path.splitext(basename)[0]}.jpg",
        )
    else:
        debug_img_path = os.path.join(
            dataset_dir,
            "queries_debug_test",
            f"debug_{os.path.splitext(basename)[0]}.jpg",
        )

    query_global = query_pose_info["global"] if query_pose_info else None

    if os.path.exists(debug_img_path):
        img_q = load_img(debug_img_path)
        query_label = f"Query (Rank {candidate['rank']} Check)"
    else:
        img_q = load_img(query_path)
        query_label = f"Query Input (No Green Box)"

    cand_abs_path = os.path.join(dataset_dir, candidate["rel_path"])
    img_c = load_img(cand_abs_path)

    target_h = 400

    def resize_h(img, h):
        aspect = img.width / img.height
        return img.resize((int(h * aspect), h), Image.Resampling.LANCZOS)

    img_q, img_c = resize_h(img_q, target_h), resize_h(img_c, target_h)

    padding, text_area_h = 10, 160
    total_w = img_q.width + img_c.width + padding * 3
    total_h = target_h + text_area_h + padding * 2

    dashboard = Image.new("RGB", (total_w, total_h), (30, 30, 30))
    draw = ImageDraw.Draw(dashboard)

    def calc_error(match):
        if query_global and match.get("global_pose"):
            dx = query_global[0] - match["global_pose"][0]
            dy = query_global[1] - match["global_pose"][1]
            return f"{math.sqrt(dx*dx + dy*dy):.2f}m"
        return "N/A"

    curr_x = padding
    dashboard.paste(img_q, (curr_x, padding + text_area_h))
    draw.text((curr_x, padding + text_area_h - 40), query_label, fill=(150, 255, 150))
    curr_x += img_q.width + padding

    dashboard.paste(img_c, (curr_x, padding + text_area_h))
    info_c = f"Rank {candidate['rank']} Candidate\nPath: {candidate.get('rel_path', 'N/A')}\nSim: {candidate.get('vpr_sim', 0):.4f}"
    if candidate.get("pose"):
        info_c += f"\nXYZ: {candidate['pose'][0]:.1f}, {candidate['pose'][1]:.1f}"
    info_c += f"\nError: {calc_error(candidate)}"
    draw.text((curr_x, padding + text_area_h - 100), info_c, fill=(200, 200, 100))

    draw.text((20, 20), f"Top-K Candidate Visualization", fill=(255, 255, 255))

    save_path = os.path.join(topk_dir, dst_name)
    dashboard.save(save_path)
    return save_path


def generate_debug_dashboard(
    query_path, match_before, match_after, result_dir, dataset_dir, query_pose_info=None
):
    def load_img(path):
        if not path or not os.path.exists(path):
            return Image.new("RGB", (300, 300), color=(50, 50, 50))
        return Image.open(path).convert("RGB")

    basename = os.path.basename(query_path)
    queries_dir = os.path.join(dataset_dir, "queries_test")

    if query_path.startswith(queries_dir):
        rel_path = os.path.relpath(query_path, queries_dir)
        debug_rel_dir = os.path.dirname(rel_path)
        debug_img_path = os.path.join(
            dataset_dir,
            "queries_debug_test",
            debug_rel_dir,
            f"debug_{os.path.splitext(basename)[0]}.jpg",
        )
    else:
        debug_img_path = os.path.join(
            dataset_dir,
            "queries_debug_test",
            f"debug_{os.path.splitext(basename)[0]}.jpg",
        )

    query_local = query_pose_info["local"] if query_pose_info else None
    query_global = query_pose_info["global"] if query_pose_info else None

    if os.path.exists(debug_img_path):
        img_q = load_img(debug_img_path)
        query_label = "Query Context (Green Box)"
    else:
        img_q = load_img(query_path)
        query_label = "Query Input (Cropped)"

    try:
        q_rel = os.path.relpath(query_path, dataset_dir)
    except:
        q_rel = basename
    query_label += f"\nPath: {q_rel}"

    if query_local:
        query_label += f"\nTrue XYZ: {query_local[0]:.1f}, {query_local[1]:.1f}"

    path_before = (
        os.path.join(dataset_dir, match_before["rel_path"]) if match_before else None
    )
    path_after = (
        os.path.join(dataset_dir, match_after["rel_path"]) if match_after else None
    )

    img_b, img_a = load_img(path_before), load_img(path_after)
    target_h = 400

    def resize_h(img, h):
        aspect = img.width / img.height
        return img.resize((int(h * aspect), h), Image.Resampling.LANCZOS)

    img_q, img_b, img_a = (
        resize_h(img_q, target_h),
        resize_h(img_b, target_h),
        resize_h(img_a, target_h),
    )

    padding, text_area_h = 10, 180
    total_w = img_q.width + img_b.width + img_a.width + padding * 4
    total_h = target_h + text_area_h + padding * 2

    dashboard = Image.new("RGB", (total_w, total_h), (30, 30, 30))
    draw = ImageDraw.Draw(dashboard)

    def calc_error(match):
        if query_global and match.get("global_pose"):
            dx = query_global[0] - match["global_pose"][0]
            dy = query_global[1] - match["global_pose"][1]
            return f"{math.sqrt(dx*dx + dy*dy):.2f}m"
        return "N/A"

    curr_x = padding
    dashboard.paste(img_q, (curr_x, padding + text_area_h))
    draw.text((curr_x, padding + text_area_h - 70), query_label, fill=(150, 255, 150))
    curr_x += img_q.width + padding

    dashboard.paste(img_b, (curr_x, padding + text_area_h))
    info_b = f"Raw Top-1 (VPR)\nPath: {match_before.get('rel_path', 'N/A')}\nSim: {match_before.get('vpr_sim', 0):.4f}"
    if match_before.get("pose"):
        info_b += f"\nXYZ: {match_before['pose'][0]:.1f}, {match_before['pose'][1]:.1f}"
    info_b += f"\nError: {calc_error(match_before)}"
    draw.text((curr_x, padding + text_area_h - 110), info_b, fill=(200, 200, 100))
    curr_x += img_b.width + padding

    dashboard.paste(img_a, (curr_x, padding + text_area_h))
    info_a = f"Final Result\nPath: {match_after.get('rel_path', 'N/A')}\nScore: {match_after.get('final_score', 0):.4f}"
    if match_after.get("pose"):
        info_a += f"\nXYZ: {match_after['pose'][0]:.1f}, {match_after['pose'][1]:.1f}"
    info_a += f"\nError: {calc_error(match_after)}"
    draw.text((curr_x, padding + text_area_h - 110), info_a, fill=(100, 255, 100))

    draw.text((20, 20), f"VPR Debug Dashboard - {basename}", fill=(255, 255, 255))
    save_path = os.path.join(result_dir, "debug_dashboard.jpg")
    dashboard.save(save_path)
    return save_path


def generate_trajectory_map(
    dataset_dir, query_pose_info, match_before, match_after, query_name, pose_manager
):
    map_dir = os.path.join(dataset_dir, "map")
    os.makedirs(map_dir, exist_ok=True)

    query_full_path = None
    for root, dirs, files in os.walk(os.path.join(dataset_dir, "queries_test")):
        for f in files:
            if f.startswith(query_name):
                query_full_path = os.path.join(root, f)
                break
        if query_full_path:
            break

    def collect_gnss_ecef(folder_name):
        pts = []
        folder_path = os.path.join(dataset_dir, folder_name)
        for root, dirs, files in os.walk(folder_path):
            if "GNSSPoses.txt" in files:
                seq_dir = root
                poses = pose_manager.load_poses(seq_dir)
                trans = pose_manager.load_transformations(seq_dir)
                if trans:
                    for ns_key, local_pose in poses.items():
                        g_pose = pose_manager.local_to_global(local_pose, trans)
                        if g_pose:
                            pts.append(g_pose)
        return pts

    db_ecef = collect_gnss_ecef("database")
    q_ecef = collect_gnss_ecef("queries_test")

    query_gnss_ecef = query_pose_info["global"] if query_pose_info else None
    mb_gnss_ecef = match_before.get("global_pose") if match_before else None
    ma_gnss_ecef = match_after.get("global_pose") if match_after else None

    try:
        ref_seq_dir = pose_manager._find_seq_dir(query_full_path)
        ref_trans = pose_manager.load_transformations(ref_seq_dir)
        ref_r = R_sci.from_quat(ref_trans["q_e_gpsw"]).inv()
        ref_t = ref_trans["t_e_gpsw"]
    except Exception:
        ref_r = R_sci.from_matrix(np.eye(3))
        ref_t = np.zeros(3)

    def ecef_to_local_2d(p_e):
        if p_e is None:
            return None
        p_local = ref_r.apply(np.array(p_e[:3]) - ref_t)
        return p_local[:2]

    db_local = [ecef_to_local_2d(p) for p in db_ecef]
    q_local = [ecef_to_local_2d(p) for p in q_ecef]

    q_marker = ecef_to_local_2d(query_gnss_ecef)
    mb_marker = ecef_to_local_2d(mb_gnss_ecef)
    ma_marker = ecef_to_local_2d(ma_gnss_ecef)

    cx, cy = 0.0, 0.0
    if q_marker is not None:
        cx, cy = q_marker[0], q_marker[1]
    elif db_local:
        cx, cy = db_local[0][0], db_local[0][1]

    fig, ax = plt.subplots(figsize=(12, 10))

    if db_local:
        xs = [p[0] - cx for p in db_local]
        ys = [p[1] - cy for p in db_local]
        ax.scatter(
            xs, ys, s=2, c="lightgray", alpha=0.8, label="Database Trajectory (RTK)"
        )
    if q_local:
        xs = [p[0] - cx for p in q_local]
        ys = [p[1] - cy for p in q_local]
        ax.scatter(
            xs, ys, s=2, c="lightblue", alpha=0.8, label="Query Trajectory (RTK)"
        )

    is_same = False
    if mb_marker is not None and ma_marker is not None:
        if math.hypot(mb_marker[0] - ma_marker[0], mb_marker[1] - ma_marker[1]) < 0.1:
            is_same = True

    if q_marker is not None and mb_marker is not None:
        ax.plot(
            [q_marker[0] - cx, mb_marker[0] - cx],
            [q_marker[1] - cy, mb_marker[1] - cy],
            "gray",
            linestyle="--",
            alpha=0.7,
        )
        dist_mb = math.hypot(q_marker[0] - mb_marker[0], q_marker[1] - mb_marker[1])
        mid_x = (q_marker[0] + mb_marker[0]) / 2 - cx
        mid_y = (q_marker[1] + mb_marker[1]) / 2 - cy

        if is_same:
            ax.text(
                mid_x,
                mid_y + 3,
                f"Err: {dist_mb:.1f}m",
                color="black",
                fontsize=12,
                fontweight="bold",
                ha="center",
                zorder=10,
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1),
            )
        else:
            ax.text(
                mid_x,
                mid_y + 3,
                f"Top-1: {dist_mb:.1f}m",
                color="darkorange",
                fontsize=10,
                fontweight="bold",
                ha="center",
                zorder=10,
            )

    if q_marker is not None and ma_marker is not None and not is_same:
        ax.plot(
            [q_marker[0] - cx, ma_marker[0] - cx],
            [q_marker[1] - cy, ma_marker[1] - cy],
            "green",
            linestyle=":",
            alpha=0.9,
        )
        dist_ma = math.hypot(q_marker[0] - ma_marker[0], q_marker[1] - ma_marker[1])
        mid_x = (q_marker[0] + ma_marker[0]) / 2 - cx
        mid_y = (q_marker[1] + ma_marker[1]) / 2 - cy
        ax.text(
            mid_x,
            mid_y - 3,
            f"Final: {dist_ma:.1f}m",
            color="green",
            fontsize=10,
            fontweight="bold",
            ha="center",
            zorder=10,
        )

    if ma_marker is not None:
        ax.scatter(
            ma_marker[0] - cx,
            ma_marker[1] - cy,
            c="lime",
            marker="s",
            s=200,
            edgecolor="black",
            label="Final Result",
            zorder=4,
        )

    if mb_marker is not None:
        ax.scatter(
            mb_marker[0] - cx,
            mb_marker[1] - cy,
            c="orange",
            marker="^",
            s=200,
            edgecolor="black",
            label="Raw Top-1",
            zorder=5,
        )

    if q_marker is not None:
        ax.scatter(
            q_marker[0] - cx,
            q_marker[1] - cy,
            c="red",
            marker="*",
            s=400,
            edgecolor="black",
            label="Query Image",
            zorder=6,
        )

    ax.set_title("VPR Trajectory Map (Absolute RTK-GNSS Ground Truth)")
    ax.set_xlabel("Relative East (m)")
    ax.set_ylabel("Relative North (m)")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_aspect("equal", adjustable="box")

    save_path = os.path.join(map_dir, f"map_gnss_{query_name}.png")
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return save_path
