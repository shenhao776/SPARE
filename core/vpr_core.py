import os
import math
import time
import json
import numpy as np
import torch
import faiss
import threading
import shutil
import cv2
from tqdm import tqdm
from types import SimpleNamespace
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import sys

from core.config import (
    CFG,
    TOP_K,
    ROOT_DIR,
    ENABLE_SL,
    SPATIAL_STEPS,
    DL_MODEL_SAVE_PATH,
    DL_HIDDEN_DIM,
    MIN_CANDIDATE_DIST,
    SUCCESS_THRESHOLD_TEST,
    FEATURE_MATCH_MIN_INLIERS,
    FEATURE_MATCH_MIN_RAW,
    RANSAC_REPROJ_THRESHOLD,
    FEATURE_HULL_MIN_INLIERS,
    vpr_logger,
    VPR_MODEL_TYPE,
    VPR_BACKBONE,
    VPR_FC_OUTPUT_DIM,
    VPR_MODEL_ID,
    TARGET_SCENE,
    STEP_FEATURE_TAU,
    STEP_MAX_DIST,
    STEP_BUDGET,
    GEOMETRY_KEYS,
    ABLATION_USE_DERIVATIVES,
    ABLATION_USE_PROGRESS,
    ABLATION_USE_SIM,
    ABLATION_USE_AREA,
    ABLATION_USE_RANK,
    ABLATION_USE_DINO,
    ABLATION_INPUT_NORM,
    ABLATION_ARCH,
    agent_input_dim,
    agent_tag,
    data_tag,
)
from core.datasets import get_adapter
from core.models import build_model, get_transform, ResizeToMultipleOf14

from core.visualizer import (
    generate_debug_dashboard,
    generate_trajectory_map,
    generate_topk_pair,
)

from sl_agent.vpr_classifier import build_sequence_features

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
extractor = None
matcher = None
batch_matcher = None
_local_models_initialized = False
_local_models_lock = threading.Lock()


def _ensure_local_models():

    global extractor, matcher, batch_matcher, _local_models_initialized
    if _local_models_initialized:
        return
    with _local_models_lock:
        if _local_models_initialized:
            return
        try:
            from lightglue import LightGlue, SuperPoint

            vpr_logger.info(
                f"⚙️ Loading SOTA models (SuperPoint + LightGlue) to {DEVICE}..."
            )
            extractor = SuperPoint(max_num_keypoints=2048).eval().to(DEVICE)
            matcher = LightGlue(features="superpoint").eval().to(DEVICE)
            batch_matcher = _build_batch_matcher(LightGlue)
        except ImportError:
            vpr_logger.error("❌ Error: lightglue library not detected.")
        _local_models_initialized = True


def _build_batch_matcher(LightGlue):

    m = LightGlue(features="superpoint", depth_confidence=-1, width_confidence=-1)
    return m.eval().to(DEVICE)


file_io_lock = threading.Lock()


def get_all_images_recursive(base_dir, valid_exts=(".png", ".jpg", ".jpeg", "tiff")):
    img_paths = []
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.lower().endswith(valid_exts):
                img_paths.append(os.path.join(root, f))
    return sorted(img_paths)


class SimpleImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, idx


def get_base_transform():

    return get_transform(CFG.model_dict)


def _hull_and_homography(hull_q, H, img_size):

    out = dict.fromkeys(GEOMETRY_KEYS, 0.0)
    if not img_size:
        return out
    w, h = float(img_size[0]), float(img_size[1])
    if w <= 0 or h <= 0:
        return out
    diag = float(np.hypot(w, h))

    m = cv2.moments(hull_q)
    if m["m00"] > 1e-9:
        cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
        out["hull_cx"] = (cx - w / 2.0) / diag
        out["hull_cy"] = (cy - h / 2.0) / diag
        _c, (rw, rh), _a = cv2.minAreaRect(hull_q)
        long_side, short_side = max(rw, rh), max(min(rw, rh), 1e-6)
        out["hull_aspect"] = float(np.tanh(np.log(long_side / short_side) / 2.0))
        rect_area = max(long_side * short_side, 1e-6)
        out["hull_fill"] = m["m00"] / rect_area - 0.5

    if H is not None and H.shape == (3, 3) and abs(H[2, 2]) > 1e-9:
        Hn = H / H[2, 2]
        if np.isfinite(Hn).all():
            det = abs(float(np.linalg.det(Hn[:2, :2])))
            if 1e-6 < det < 1e6:
                out["H_tx"] = float(np.tanh(Hn[0, 2] / diag))
                out["H_ty"] = float(np.tanh(Hn[1, 2] / diag))
                out["H_logdet"] = float(np.tanh(np.log(np.sqrt(det))))
                out["H_persp"] = float(
                    np.tanh(np.log1p(np.hypot(Hn[2, 0], Hn[2, 1]) * diag))
                )
    return {k: (float(v) if np.isfinite(v) else 0.0) for k, v in out.items()}


ONLINE_DEFAULT_ZOOM = 3.2


_WALK_TABLES = {}


def _walk_table(db_full_path):

    parts = os.path.realpath(db_full_path).split(os.sep)
    try:
        scene = parts[parts.index("datasets") + 1]
    except (ValueError, IndexError):
        raise RuntimeError(
            f"step_feature_tau is set but the scene cannot be read off "
            f"{db_full_path!r} (expected .../datasets/<Scene>/...)"
        )
    if scene not in _WALK_TABLES:
        path = os.path.join(
            ROOT_DIR, "assets", "map_feature_rate", f"{scene}_walk.json"
        )
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"step_feature_tau is set but {path} is missing. Build it with:\n"
                f"  python tools/map_feature_rate.py --scene {scene} --stride 1\n"
                f"  python tools/map_feature_rate.py --scene {scene} --stride 3\n"
                f"  python tools/map_feature_rate.py --scene {scene} --emit-walk"
            )
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        _WALK_TABLES[scene] = (payload["pairs"], set(payload.get("breaks") or []))
    return _WALK_TABLES[scene]


def db_features_cache_valid(descriptors, paths, abs_db_paths):

    if not (
        getattr(descriptors, "ndim", 0) == 2
        and len(paths) > 0
        and descriptors.shape[0] == len(paths)
    ):
        return False
    if len(paths) != len(abs_db_paths):
        return False
    return sorted(str(p) for p in paths) == sorted(str(p) for p in abs_db_paths)


def get_database_spatial_sequence(
    db_full_path,
    pose_mgr,
    spatial_steps,
    step_feature_tau=STEP_FEATURE_TAU,
    step_max_dist=STEP_MAX_DIST,
    step_budget=STEP_BUDGET,
    return_meta=False,
):

    back_count, fwd_count = spatial_steps
    if back_count == 0 and fwd_count == 0:
        if return_meta:
            return [], [], [{"change": 0.0, "dist": 0.0, "valid": True}]
        return [], []
    seq_dir = os.path.dirname(db_full_path)
    all_files = [
        f
        for f in os.listdir(seq_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", "tiff"))
    ]

    def get_ts(filename):
        try:
            return int(filename.split(".")[0].split("@")[0])
        except:
            return 0

    all_files.sort(key=get_ts)
    target_name = os.path.basename(db_full_path)
    if target_name not in all_files:
        return [db_full_path] * back_count, [db_full_path] * fwd_count

    idx = all_files.index(target_name)
    target_info = pose_mgr.get_pose_info(db_full_path)
    if not target_info or not target_info.get("global"):
        return [db_full_path] * back_count, [db_full_path] * fwd_count

    candidate_global = target_info["global"]

    feature_driven = step_feature_tau is not None
    pairs, breaks = _walk_table(db_full_path) if feature_driven else ({}, set())

    def pair_change(lo, hi):

        key = f"{all_files[lo]}|{all_files[hi]}"
        change = pairs.get(key)
        if change is None:
            raise KeyError(
                f"walk table has no entry for adjacent pair {key!r} -- the "
                f"datasets/ listing has drifted since the table was built. "
                f"Rebuild it: python tools/map_feature_rate.py --scene <Scene> "
                f"--stride 1, --stride 3, then --emit-walk"
            )
        return change

    def is_break(lo, hi):
        return f"{all_files[lo]}|{all_files[hi]}" in breaks

    def walk(indices, adjacent, count):

        paths, meta = [], []
        last_global = candidate_global
        acc = 0.0
        total = 0.0
        for i in indices:
            if feature_driven:
                lo, hi = adjacent(i)
                if is_break(lo, hi):
                    break
                step = pair_change(lo, hi)
                acc += step
                total += step
            cand_path = os.path.join(seq_dir, all_files[i])
            info = pose_mgr.get_pose_info(cand_path)
            if not info or not info.get("global"):
                continue
            cand_g = info["global"]

            dist = pose_mgr.calculate_physical_distance(
                db_full_path, last_global, cand_g
            )

            adopt = acc >= step_feature_tau or (
                step_max_dist is not None and dist is not None and dist >= step_max_dist
            )

            if adopt:
                paths.append(cand_path)
                meta.append({"change": acc, "dist": dist, "valid": True})
                last_global = cand_g
                acc = 0.0

            if len(paths) >= count:
                break
            if feature_driven and step_budget is not None and total >= step_budget:
                break

        while len(paths) < count:
            paths.append(paths[-1] if paths else db_full_path)
            meta.append({"change": 0.0, "dist": 0.0, "valid": False})
        return paths, meta

    history_paths, back_meta = walk(
        range(idx - 1, -1, -1), lambda i: (i, i + 1), back_count
    )
    future_paths, fwd_meta = walk(
        range(idx + 1, len(all_files)), lambda i: (i - 1, i), fwd_count
    )

    if return_meta:
        centre = {"change": 0.0, "dist": 0.0, "valid": True}
        return (
            history_paths,
            future_paths,
            list(reversed(back_meta)) + [centre] + fwd_meta,
        )
    return history_paths, future_paths


class PoseManager:

    def __init__(self, pose_format=None):
        self.adapter = get_adapter(pose_format or CFG.dataset.pose_format)
        self.cache = {}
        self.seq_dir_cache = {}
        self.keys_cache = {}
        self.img_pose_cache = {}

    def _find_seq_dir(self, img_abs_path):
        img_dir = os.path.dirname(os.path.abspath(img_abs_path))
        if img_dir in self.seq_dir_cache:
            return self.seq_dir_cache[img_dir]
        curr_dir = img_dir
        while True:
            if (
                os.path.exists(os.path.join(curr_dir, "undistorted_images"))
                or os.path.exists(os.path.join(curr_dir, "Transformations.txt"))
                or os.path.exists(os.path.join(curr_dir, "GNSSPoses.txt"))
                or os.path.exists(os.path.join(curr_dir, "pose.txt"))
            ):
                self.seq_dir_cache[img_dir] = curr_dir
                return curr_dir
            parent_dir = os.path.dirname(curr_dir)
            if parent_dir == curr_dir:
                break
            curr_dir = parent_dir
        raise FileNotFoundError(f"Cannot find sequence root directory: {img_abs_path}")

    def load_poses(self, sequence_dir):

        return self.adapter.load_poses(sequence_dir)

    def load_transformations(self, sequence_dir):
        fn = getattr(self.adapter, "load_transformations", None)
        return fn(sequence_dir) if fn else None

    def local_to_global(self, local_pose, trans):
        fn = getattr(self.adapter, "local_to_global", None)
        return fn(local_pose, trans) if fn else None

    def get_pose_info(self, img_abs_path):
        if not img_abs_path:
            return None
        if img_abs_path in self.img_pose_cache:
            return self.img_pose_cache[img_abs_path]

        try:
            seq_dir = self._find_seq_dir(img_abs_path)
        except:
            return None

        if seq_dir not in self.cache:
            poses = self.adapter.load_poses(seq_dir)
            self.cache[seq_dir] = poses
            self.keys_cache[seq_dir] = np.array(list(poses.keys()))

        local_pose = None
        try:
            filename = os.path.basename(img_abs_path)
            ts_str = os.path.splitext(filename)[0].split("@")[0]
            if "." in ts_str:
                from decimal import Decimal

                img_ns = int(Decimal(ts_str) * Decimal("1e9"))
            else:
                val = int(ts_str)
                if val in self.cache[seq_dir]:
                    img_ns = val
                elif len(ts_str) == 10:
                    img_ns = val * 1_000_000_000
                elif len(ts_str) == 13:
                    img_ns = val * 1_000_000
                elif len(ts_str) == 16:
                    img_ns = val * 1_000
                else:
                    img_ns = val

            if img_ns in self.cache[seq_dir]:
                local_pose = self.cache[seq_dir][img_ns]
            else:
                keys = self.keys_cache[seq_dir]
                if len(keys) > 0:
                    diffs = np.abs(keys - img_ns)
                    idx = diffs.argmin()
                    if diffs[idx] < 250000000:
                        local_pose = self.cache[seq_dir][keys[idx]]
        except:
            pass

        result = None
        if local_pose:
            result = {
                "local": local_pose[:3],
                "global": self.adapter.to_global(local_pose, seq_dir),
            }

        self.img_pose_cache[img_abs_path] = result
        return result

    def calculate_physical_distance(self, ref_img_path, global_A, global_B):
        if not global_A or not global_B:
            return None
        try:
            seq_dir = self._find_seq_dir(ref_img_path)
            return self.adapter.physical_distance(global_A, global_B, seq_dir)
        except Exception:
            return math.dist(global_A[:3], global_B[:3])

    def calculate_angular_distance(self, global_A, global_B):
        return self.adapter.angular_distance(global_A, global_B)


pose_manager = PoseManager()


class DynamicCropProjector:
    def __init__(self):
        pass

    def unpack_offline_features(self, step_feats, device):
        step_feats_gpu = {}
        for k, v in step_feats.items():
            if isinstance(v, list):
                step_feats_gpu[k] = [
                    (
                        t.to(device).float()
                        if isinstance(t, torch.Tensor) and t.is_floating_point()
                        else (t.to(device) if isinstance(t, torch.Tensor) else t)
                    )
                    for t in v
                ]
            else:
                step_feats_gpu[k] = (
                    v.to(device).float()
                    if isinstance(v, torch.Tensor) and v.is_floating_point()
                    else (v.to(device) if isinstance(v, torch.Tensor) else v)
                )
        return step_feats_gpu

    def pack_offline_features(self, feats):
        feats_cpu = {}
        if feats is not None:
            for k, v in feats.items():
                if k == "descriptors":
                    if isinstance(v, list):
                        feats_cpu[k] = [
                            (t.cpu().half() if isinstance(t, torch.Tensor) else t.cpu())
                            for t in v
                        ]
                    else:
                        feats_cpu[k] = (
                            v.cpu().half() if isinstance(v, torch.Tensor) else v.cpu()
                        )
                else:
                    if isinstance(v, list):
                        feats_cpu[k] = [
                            t.cpu() if isinstance(t, torch.Tensor) else t for t in v
                        ]
                    else:
                        feats_cpu[k] = v.cpu() if isinstance(v, torch.Tensor) else v
        return feats_cpu

    def compute_geometric_metrics(
        self, kp_q, kp_s, matches, with_geometry=False, img_size=None
    ):

        valid_inliers, scale_ratio, kp_area_s, kp_area_q = 0, 0.0, 0.0, 0.0
        geo = dict.fromkeys(GEOMETRY_KEYS, 0.0)
        if len(matches) >= FEATURE_MATCH_MIN_INLIERS:
            src_pts = np.float32([kp_s[m.trainIdx].pt for m in matches]).reshape(
                -1, 1, 2
            )
            dst_pts = np.float32([kp_q[m.queryIdx].pt for m in matches]).reshape(
                -1, 1, 2
            )
            cv2.setRNGSeed(0)
            H, mask = cv2.findHomography(
                src_pts, dst_pts, cv2.RANSAC, RANSAC_REPROJ_THRESHOLD
            )
            if mask is not None:
                valid_inliers = int(np.sum(mask))
                if valid_inliers >= FEATURE_HULL_MIN_INLIERS:
                    valid_matches = [
                        m for m_idx, m in enumerate(matches) if mask[m_idx][0] == 1
                    ]
                    hull_s = cv2.convexHull(
                        np.float32([kp_s[m.trainIdx].pt for m in valid_matches])
                    )
                    hull_q = cv2.convexHull(
                        np.float32([kp_q[m.queryIdx].pt for m in valid_matches])
                    )
                    kp_area_s = cv2.contourArea(hull_s)
                    kp_area_q = cv2.contourArea(hull_q)
                    if kp_area_q > 0:
                        scale_ratio = kp_area_s / kp_area_q
                    if with_geometry:
                        geo = _hull_and_homography(hull_q, H, img_size)
        if with_geometry:
            return valid_inliers, scale_ratio, kp_area_s, kp_area_q, geo
        return valid_inliers, scale_ratio, kp_area_s, kp_area_q

    def extract_image(self, img_cv, image_path=None, dataset_path=None):
        if img_cv is None:
            return None
        _ensure_local_models()
        if extractor is None:
            return None
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        t_img = (
            torch.from_numpy(img_rgb).float().permute(2, 0, 1).unsqueeze(0).to(DEVICE)
            / 255.0
        )
        with torch.no_grad():
            feats = extractor.extract(t_img)
        return feats

    def match_features(self, feats_curr, feats_prev):
        if not feats_curr or not feats_prev:
            return [], [], []
        for feats in (feats_curr, feats_prev):
            kpts = feats.get("keypoints") if hasattr(feats, "get") else None
            if kpts is None or kpts.shape[-2] < FEATURE_MATCH_MIN_RAW:
                return [], [], []
        _ensure_local_models()
        if extractor is None or matcher is None:
            return [], [], []
        with torch.no_grad():
            matches_dict = matcher({"image0": feats_curr, "image1": feats_prev})
        matches = matches_dict["matches"][0]
        if len(matches) < FEATURE_MATCH_MIN_RAW:
            return [], [], []
        pts_curr = feats_curr["keypoints"][0][matches[..., 0]].cpu().numpy()
        pts_prev = feats_prev["keypoints"][0][matches[..., 1]].cpu().numpy()
        kp_curr = [
            cv2.KeyPoint(x=float(p[0]), y=float(p[1]), size=1.0) for p in pts_curr
        ]
        kp_prev = [
            cv2.KeyPoint(x=float(p[0]), y=float(p[1]), size=1.0) for p in pts_prev
        ]
        good_matches = [
            cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=0.0)
            for i in range(len(pts_curr))
        ]
        return kp_curr, kp_prev, good_matches

    def match_features_batch(self, feats_curr, feats_list, max_batch=8):

        n = len(feats_list)
        out = [([], [], []) for _ in range(n)]
        if not feats_curr or n == 0:
            return out
        _ensure_local_models()
        if extractor is None or batch_matcher is None:
            return out

        groups = {}
        for i, f in enumerate(feats_list):
            if not f:
                continue
            kp = f.get("keypoints")
            if kp is None or kp.shape[-2] == 0:
                continue
            groups.setdefault(int(kp.shape[-2]), []).append(i)

        for _kn, idxs in groups.items():
            for s in range(0, len(idxs), max_batch):
                chunk = idxs[s : s + max_batch]
                b = len(chunk)
                data1 = {
                    k: torch.cat([feats_list[i][k] for i in chunk], dim=0)
                    for k in ("keypoints", "descriptors", "image_size")
                    if feats_list[chunk[0]].get(k) is not None
                }
                data0 = {
                    k: v.expand(b, *v.shape[1:])
                    for k, v in feats_curr.items()
                    if isinstance(v, torch.Tensor) and v.dim() >= 2
                }
                with torch.no_grad():
                    res = batch_matcher({"image0": data0, "image1": data1})
                for j, i in enumerate(chunk):
                    out[i] = self._unpack_pair(
                        res["matches"][j], feats_curr, feats_list[i]
                    )
        return out

    @staticmethod
    def _unpack_pair(matches, feats_q, feats_s):

        if matches is None or len(matches) < FEATURE_MATCH_MIN_RAW:
            return [], [], []
        pts_q = feats_q["keypoints"][0][matches[..., 0]].cpu().numpy()
        pts_s = feats_s["keypoints"][0][matches[..., 1]].cpu().numpy()
        kp_q = [cv2.KeyPoint(x=float(p[0]), y=float(p[1]), size=1.0) for p in pts_q]
        kp_s = [cv2.KeyPoint(x=float(p[0]), y=float(p[1]), size=1.0) for p in pts_s]
        good = [
            cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=0.0)
            for i in range(len(pts_q))
        ]
        return kp_q, kp_s, good

    def is_valid_projection(self, corners, img_w, img_h, original_area):
        if not cv2.isContourConvex(np.int32(corners)):
            return False, "Perspective collapse"
        area_oriented = cv2.contourArea(np.float32(corners), oriented=True)
        if area_oriented < 0:
            return False, "Mirror flip detected"
        projected_area = abs(area_oriented)
        if projected_area < 50 or projected_area > original_area * 6:
            return False, "Area anomaly"
        xs = corners[:, 0, 0]
        ys = corners[:, 0, 1]
        bbox_w = np.max(xs) - np.min(xs)
        bbox_h = np.max(ys) - np.min(ys)
        if bbox_w > img_w * 4 or bbox_h > img_h * 4:
            return False, "Perspective stretch overflow"
        if bbox_w * bbox_h > projected_area * 10:
            return False, "Severe shear stretch"
        return True, "Valid"

    def process_sequence(
        self, candidate_img_path, history_paths, future_paths, save_dir=None
    ):
        img_candidate = cv2.cvtColor(
            np.array(Image.open(candidate_img_path).convert("RGB")), cv2.COLOR_RGB2BGR
        )
        img_target_rgb = cv2.cvtColor(img_candidate, cv2.COLOR_BGR2RGB)
        img_h, img_w = img_candidate.shape[:2]
        original_area = img_w * img_h
        original_corners = np.float32(
            [[0, 0], [img_w, 0], [img_w, img_h], [0, img_h]]
        ).reshape(-1, 1, 2)

        def _track_branch(branch_paths, branch_name):
            crops = []
            H_global = np.eye(3, dtype=np.float32)
            feats_prev = self.extract_image(
                img_candidate, image_path=candidate_img_path
            )
            chain_broken = False
            last_valid_crop_rgb = img_target_rgb.copy()
            last_valid_combined = None

            if save_dir:
                os.makedirs(save_dir, exist_ok=True)

            for idx, curr_path in enumerate(branch_paths):
                img_curr = cv2.cvtColor(
                    np.array(Image.open(curr_path).convert("RGB")), cv2.COLOR_RGB2BGR
                )
                img_curr_with_box = img_curr.copy()
                is_out_of_view = False
                feats_curr = self.extract_image(img_curr, image_path=curr_path)

                if not chain_broken:
                    kp_curr, kp_prev, matches = self.match_features(
                        feats_curr, feats_prev
                    )
                else:
                    kp_curr, kp_prev, matches = [], [], []

                if chain_broken:
                    is_out_of_view = True
                elif len(matches) < FEATURE_MATCH_MIN_RAW:
                    is_out_of_view = True
                    chain_broken = True
                else:
                    src_pts = np.float32(
                        [kp_prev[m.trainIdx].pt for m in matches]
                    ).reshape(-1, 1, 2)
                    dst_pts = np.float32(
                        [kp_curr[m.queryIdx].pt for m in matches]
                    ).reshape(-1, 1, 2)
                    cv2.setRNGSeed(0)
                    H_step, mask_homo = cv2.findHomography(
                        src_pts, dst_pts, cv2.RANSAC, RANSAC_REPROJ_THRESHOLD
                    )
                    is_valid = False

                    if H_step is not None:
                        H_global_tentative = np.matmul(H_step, H_global)
                        projected_corners = cv2.perspectiveTransform(
                            original_corners, H_global_tentative
                        )
                        is_valid, _ = self.is_valid_projection(
                            projected_corners, img_w, img_h, original_area
                        )

                    if is_valid:
                        H_global = H_global_tentative
                    else:
                        cv2.setRNGSeed(0)
                        Affine_step, mask_affine = cv2.estimateAffinePartial2D(
                            src_pts,
                            dst_pts,
                            method=cv2.RANSAC,
                            ransacReprojThreshold=5.0,
                        )
                        if Affine_step is not None:
                            H_affine = np.eye(3, dtype=np.float32)
                            H_affine[0:2, :] = Affine_step
                            H_global_tentative = np.matmul(H_affine, H_global)
                            projected_corners = cv2.perspectiveTransform(
                                original_corners, H_global_tentative
                            )
                            is_valid, _ = self.is_valid_projection(
                                projected_corners, img_w, img_h, original_area
                            )
                            if is_valid:
                                H_global = H_global_tentative
                            else:
                                is_out_of_view = True
                                chain_broken = True
                        else:
                            is_out_of_view = True
                            chain_broken = True

                projected_corners = cv2.perspectiveTransform(original_corners, H_global)
                bounding_rect = cv2.boundingRect(np.int32(projected_corners))
                bx, by, bw, bh = bounding_rect
                c_x1, c_y1 = max(0, bx), max(0, by)
                c_x2, c_y2 = min(img_w, bx + bw), min(img_h, by + bh)
                intersect_w = max(0, c_x2 - c_x1)
                intersect_h = max(0, c_y2 - c_y1)

                if not is_out_of_view:
                    if intersect_w < 200 or intersect_h < 200:
                        is_out_of_view = True
                        chain_broken = True
                    else:
                        cropped_roi_bgr = img_curr[c_y1:c_y2, c_x1:c_x2]
                        cropped_roi_rgb = cv2.cvtColor(
                            cropped_roi_bgr, cv2.COLOR_BGR2RGB
                        )
                        last_valid_crop_rgb = cropped_roi_rgb.copy()

                        if save_dir:
                            cv2.polylines(
                                img_curr_with_box,
                                [np.int32(projected_corners)],
                                True,
                                (0, 255, 0),
                                3,
                            )
                            cv2.rectangle(
                                img_curr_with_box,
                                (c_x1, c_y1),
                                (c_x2, c_y2),
                                (0, 0, 255),
                                2,
                            )
                            h, w = img_curr_with_box.shape[:2]
                            aspect_ratio = (
                                cropped_roi_bgr.shape[1] / cropped_roi_bgr.shape[0]
                            )
                            new_width = int(h * aspect_ratio)
                            cropped_resized = cv2.resize(
                                cropped_roi_bgr, (new_width, h)
                            )
                            combined_img = cv2.hconcat(
                                [img_curr_with_box, cropped_resized]
                            )

                            c_h, c_w = combined_img.shape[:2]
                            if c_h > 360:
                                scale = 360.0 / c_h
                                combined_img = cv2.resize(
                                    combined_img, (int(c_w * scale), int(c_h * scale))
                                )

                            last_valid_combined = combined_img.copy()
                            cv2.imwrite(
                                os.path.join(
                                    save_dir,
                                    f"{branch_name}_step_{idx+1:02d}_combined.jpg",
                                ),
                                combined_img,
                                [int(cv2.IMWRITE_JPEG_QUALITY), 50],
                            )

                if is_out_of_view:
                    cropped_roi_rgb = last_valid_crop_rgb.copy()
                    if save_dir and last_valid_combined is not None:
                        fallback_vis = last_valid_combined.copy()
                        cv2.putText(
                            fallback_vis,
                            "FALLBACK: OUT OF VIEW",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 0, 255),
                            2,
                        )
                        cv2.imwrite(
                            os.path.join(
                                save_dir,
                                f"{branch_name}_step_{idx+1:02d}_combined_fallback.jpg",
                            ),
                            fallback_vis,
                            [int(cv2.IMWRITE_JPEG_QUALITY), 50],
                        )

                crops.append(Image.fromarray(cropped_roi_rgb))
                if not chain_broken:
                    feats_prev = feats_curr

            return crops

        history_crops = _track_branch(history_paths, "History") if history_paths else []
        future_crops = _track_branch(future_paths, "Future") if future_paths else []
        result_seq = list(reversed(history_crops))
        result_seq.append(Image.fromarray(img_target_rgb))
        result_seq.extend(future_crops)
        return result_seq


class VPRService:
    def __init__(self, fc_output_dim=VPR_FC_OUTPUT_DIM):
        vpr_logger.info(f"[SERVER] Initializing {VPR_MODEL_TYPE} Engine...")
        self.lock = threading.Lock()
        self.infer_lock = threading.Lock()

        self.args = SimpleNamespace(
            fc_output_dim=fc_output_dim,
            infer_batch_size=4,
            device="cuda" if torch.cuda.is_available() else "cpu",
            num_workers=4,
        )

        self.model = build_model(CFG.model_dict, self.args.device)

        self.db_cache = {}
        self.cropper = DynamicCropProjector()
        self.state_builder = None

        try:
            from sl_agent.vpr_classifier import VPRReranking

            dynamic_input_dim = agent_input_dim()
            ablation_str = agent_tag()
            scene_suffix = f"_{TARGET_SCENE}" if TARGET_SCENE else "_ALL_SCENES"
            ablation_model_path = DL_MODEL_SAVE_PATH.replace(
                ".pth", f"_{ablation_str}{scene_suffix}{data_tag()}.pth"
            )

            legacy_model_path = DL_MODEL_SAVE_PATH
            if os.getenv("USE_MASTER_WEIGHT") == "True":
                ablation_model_path = ablation_model_path.replace(
                    VPR_MODEL_ID, "MegaLoc_8448"
                )
                legacy_model_path = legacy_model_path.replace(
                    VPR_MODEL_ID, "MegaLoc_8448"
                )

            model_candidates = [ablation_model_path, legacy_model_path]
            model_candidates = list(dict.fromkeys(model_candidates))
            model_save_path = next(
                (p for p in model_candidates if os.path.exists(p)), None
            )
            explicit_weight = os.getenv("STAR_RERANK_WEIGHT")
            if explicit_weight:
                if not os.path.isfile(explicit_weight):
                    raise FileNotFoundError(
                        f"STAR_RERANK_WEIGHT not found: {explicit_weight}"
                    )
                model_save_path = explicit_weight
            explicit_weight = os.getenv("STAR_RERANK_WEIGHT")
            if explicit_weight:
                if not os.path.isfile(explicit_weight):
                    raise FileNotFoundError(
                        f"STAR_RERANK_WEIGHT not found: {explicit_weight}"
                    )
                model_save_path = explicit_weight

            if model_save_path is not None and ENABLE_SL:
                self.reranking_model = VPRReranking(
                    input_dim=dynamic_input_dim,
                    hidden_dim=DL_HIDDEN_DIM,
                    arch=ABLATION_ARCH,
                    input_norm=ABLATION_INPUT_NORM,
                ).to(self.args.device)
                self.reranking_model.load_state_dict(
                    torch.load(
                        model_save_path,
                        map_location=self.args.device,
                        weights_only=True,
                    )
                )
                self.reranking_model.eval()
                vpr_logger.info(
                    f"[SERVER] 🤖 Reranking Agent Loaded for {VPR_MODEL_ID}: {os.path.basename(model_save_path)}"
                )
            else:
                self.reranking_model = None
                vpr_logger.info(
                    f"[SERVER] Reranking Agent Disabled or Not Found for {VPR_MODEL_ID}. Tried: {model_candidates}"
                )
        except Exception as e:
            self.reranking_model = None
            vpr_logger.info(f"[SERVER] Live re-ranking not available: {e}")

    def _get_or_compute_db_features(self, dataset_folder):
        with self.lock:
            scene_suffix = f"_{TARGET_SCENE}" if TARGET_SCENE else "_ALL_SCENES"
            cache_key = f"{dataset_folder}_{VPR_MODEL_ID}{scene_suffix}_features"
            if cache_key in self.db_cache:
                return self.db_cache[cache_key]

            asset_base_dir = os.path.join("assets", VPR_MODEL_ID)
            os.makedirs(asset_base_dir, exist_ok=True)
            features_cache_file = os.path.join(
                asset_base_dir, f"database_features{scene_suffix}.npz"
            )
            db_dir = os.path.join(dataset_folder, "database")
            if not os.path.isdir(db_dir):
                raise FileNotFoundError(
                    f"Database directory not found: {db_dir}. "
                    "Please build dataset first (e.g., tools/build_paper_dataset.py)."
                )

            abs_db_paths = get_all_images_recursive(db_dir)
            if len(abs_db_paths) == 0:
                raise ValueError(
                    f"No database images found in: {db_dir}. "
                    "Please check dataset structure and rebuild dataset if needed."
                )

            if os.path.exists(features_cache_file):
                try:
                    with np.load(features_cache_file, allow_pickle=True) as cache_data:
                        descriptors = cache_data["descriptors"]
                        paths = cache_data["paths"]

                    is_valid = db_features_cache_valid(
                        descriptors,
                        paths,
                        [os.path.relpath(p, dataset_folder) for p in abs_db_paths],
                    )
                    if is_valid:
                        self.db_cache[cache_key] = (descriptors, paths)
                        return self.db_cache[cache_key]

                    vpr_logger.warning(
                        f"⚠️ DB cache is empty or structurally invalid: "
                        f"{features_cache_file}. Rebuilding..."
                    )
                    os.remove(features_cache_file)
                except Exception as e:
                    vpr_logger.warning(
                        f"⚠️ Failed to read DB cache ({features_cache_file}): {e}. Rebuilding cache..."
                    )
                    try:
                        os.remove(features_cache_file)
                    except Exception:
                        pass

            rel_db_paths = np.array(
                [os.path.relpath(p, dataset_folder) for p in abs_db_paths]
            )

            db_batch = self.args.infer_batch_size
            if db_batch > 1:
                seen = set()
                for _p in abs_db_paths:
                    try:
                        with Image.open(_p) as _im:
                            seen.add(_im.size)
                    except Exception:
                        seen.add(None)
                    if len(seen) > 1:
                        break
                if len(seen) > 1:
                    vpr_logger.warning(
                        f"⚠️ 库图像尺寸不一({sorted(map(str, seen))} …),"
                        f"描述子提取改为 batch_size=1"
                    )
                    db_batch = 1
            loader = DataLoader(
                SimpleImageDataset(abs_db_paths, get_base_transform()),
                batch_size=db_batch,
                num_workers=self.args.num_workers,
            )
            if hasattr(self.model, "build_vocab"):
                self.model.build_vocab(loader, db_path=db_dir)

            descriptors = np.empty(
                (len(abs_db_paths), self.args.fc_output_dim), dtype="float32"
            )
            with torch.no_grad():
                for i, (images, indices) in enumerate(
                    tqdm(
                        loader,
                        desc=f"{VPR_MODEL_TYPE} Encoding ({TARGET_SCENE or 'ALL'})",
                        mininterval=20,
                    )
                ):
                    desc = self.model(images.to(self.args.device))
                    descriptors[indices.numpy(), :] = desc.cpu().numpy()

            temporary_npz = f"{features_cache_file}.tmp.{os.getpid()}.npz"
            np.savez(temporary_npz, descriptors=descriptors, paths=rel_db_paths)
            os.replace(temporary_npz, features_cache_file)
            self.db_cache[cache_key] = (descriptors, rel_db_paths)
            return descriptors, rel_db_paths

    def run_inference(self, app_args):
        with self.infer_lock:
            inference_start = time.perf_counter()
            try:
                query_name = app_args.query_image_filename.split(".")[0]
                is_batch = getattr(app_args, "is_batch", False)

                result_dir = os.path.join(app_args.dataset_path, "result", query_name)
                topk_dir = os.path.join(result_dir, "top-k")

                if not is_batch:
                    if os.path.exists(topk_dir):
                        shutil.rmtree(topk_dir)
                    os.makedirs(topk_dir, exist_ok=True)

                db_desc, db_rel_paths = self._get_or_compute_db_features(
                    app_args.dataset_path
                )
                queries_dir = os.path.join(app_args.dataset_path, "queries_test")
                all_q_paths = get_all_images_recursive(queries_dir)
                query_full_path = next(
                    (
                        q
                        for q in all_q_paths
                        if os.path.basename(q).startswith(query_name)
                    ),
                    None,
                )

                if not query_full_path:
                    raise FileNotFoundError(f"Query {query_name} not found.")

                if not is_batch:
                    vpr_logger.info("\n" + "=" * 65)
                    vpr_logger.info(
                        f"🚀 [SPARET QUERY] Processing query frame: {query_name}"
                    )
                    vpr_logger.info("=" * 65)
                else:
                    vpr_logger.debug(
                        f"🚀 [SPARET QUERY] Batch processing: {query_name}"
                    )

                query_pose_info = pose_manager.get_pose_info(query_full_path)
                query_pose = query_pose_info["local"] if query_pose_info else None
                query_global_pose = (
                    query_pose_info["global"] if query_pose_info else None
                )

                transform = get_base_transform()
                img_tensor = (
                    transform(Image.open(query_full_path).convert("RGB"))
                    .unsqueeze(0)
                    .to(self.args.device)
                )

                with torch.no_grad():
                    q_desc = self.model(img_tensor).cpu().numpy()
                q_norm = np.linalg.norm(q_desc[0])

                index = faiss.IndexFlatL2(self.args.fc_output_dim)
                index.add(db_desc)

                target_k = TOP_K
                search_k = min(len(db_rel_paths), target_k * 10)

                candidates = []
                selected_globals = []
                skipped_candidates = []

                while len(candidates) < target_k:
                    distances, predictions = index.search(q_desc, search_k)
                    candidates, selected_globals, skipped_candidates = [], [], []

                    for db_idx in predictions[0]:
                        rel_p = db_rel_paths[db_idx]
                        abs_p = os.path.join(app_args.dataset_path, rel_p)
                        db_pose_info = pose_manager.get_pose_info(abs_p)
                        p = db_pose_info["local"] if db_pose_info else None
                        p_global = db_pose_info["global"] if db_pose_info else None

                        db_feat = db_desc[db_idx]
                        cos_sim = np.dot(db_feat, q_desc[0]) / (
                            np.linalg.norm(db_feat) * q_norm
                        )
                        sim = max(0.0, float(cos_sim))

                        cand_error = None
                        if query_global_pose and p_global:
                            cand_error = pose_manager.calculate_physical_distance(
                                query_full_path, query_global_pose, p_global
                            )

                        cand_info = {
                            "db_idx": db_idx,
                            "rel_path": rel_p,
                            "basename": os.path.basename(rel_p),
                            "vpr_sim": sim,
                            "pose": p,
                            "global_pose": p_global,
                            "error": cand_error,
                        }

                        if p_global is not None:
                            is_too_close = False
                            for sg in selected_globals:
                                dist = pose_manager.calculate_physical_distance(
                                    query_full_path, p_global, sg
                                )
                                if dist is not None and dist < MIN_CANDIDATE_DIST:
                                    is_too_close = True
                                    break
                            if is_too_close:
                                skipped_candidates.append(cand_info)
                                continue
                            selected_globals.append(p_global)

                        candidates.append(cand_info)
                        if len(candidates) >= target_k:
                            break

                    if len(candidates) >= target_k or search_k >= len(db_rel_paths):
                        break
                    search_k = min(len(db_rel_paths), search_k * 2)

                if len(candidates) < target_k:
                    needed = target_k - len(candidates)
                    candidates.extend(skipped_candidates[:needed])

                for idx, cand in enumerate(candidates):
                    cand["rank"] = idx + 1

                best_match = None

                if (
                    ENABLE_SL
                    and self.reranking_model is not None
                    and getattr(app_args, "rerank", True)
                ):
                    from sl_agent.online_state import OnlineStateBuilder

                    max_steps = sum(SPATIAL_STEPS) + 1
                    query_cv_img = cv2.imread(query_full_path)
                    if self.state_builder is None:
                        self.state_builder = OnlineStateBuilder(
                            self, app_args.dataset_path
                        )

                    vpr_logger.info("  >>> Building the online state...")
                    db_lookup = {rel: db_desc[i] for i, rel in enumerate(db_rel_paths)}
                    rows = self.state_builder.build_rows(
                        query_cv_img,
                        query_full_path,
                        candidates,
                        db_lookup,
                        q_desc[0],
                        zoom=getattr(app_args, "zoom", ONLINE_DEFAULT_ZOOM),
                    )
                    x_tensor, mask_tensor = self.state_builder.build_state_tensor(rows)
                    valid_cands = candidates

                    best_match_candidate = candidates[0]
                    best_prob = 0.0

                    for t in range(1, x_tensor.shape[2] + 1):
                        step_x = x_tensor[:, :, :t, :]
                        with torch.no_grad():
                            probs = self.reranking_model.predict_softmax(
                                step_x, mask=mask_tensor
                            )
                            curr_probs = probs[0, :, -1]
                            max_p, max_idx = torch.max(curr_probs, dim=0)

                        if max_idx.item() < len(valid_cands):
                            cand_name = valid_cands[max_idx.item()]["basename"]
                            vpr_logger.debug(
                                f"      ↳ Step {t}/{max_steps}: current leader {cand_name} (p={max_p.item():.2f})"
                            )

                    with torch.no_grad():
                        probs = self.reranking_model.predict_softmax(
                            x_tensor, mask=mask_tensor
                        )
                        curr_probs = probs[0, :, -1]
                        max_p, max_idx = torch.max(curr_probs, dim=0)

                    if max_idx.item() < len(valid_cands):
                        gate = float(os.getenv("STAR_GATE_MARGIN", "0"))
                        margin = float(curr_probs[max_idx] - curr_probs[0])
                        if margin >= gate:
                            best_match_candidate = valid_cands[max_idx.item()]
                            best_prob = max_p.item()
                    vpr_logger.info(
                        f"  └─ 🏁 [Final rollout] Rank {best_match_candidate['rank']} wins with p={best_prob:.2f}."
                    )

                    best_match = best_match_candidate

                if best_match is None:
                    best_match = candidates[0]
                best_match["final_score"] = best_match["vpr_sim"]

                distance_error = None
                if query_global_pose and best_match.get("global_pose"):
                    distance_error = pose_manager.calculate_physical_distance(
                        query_full_path,
                        query_global_pose,
                        best_match.get("global_pose"),
                    )

                db_path, map_path = "", ""

                db_path = None
                map_path = None
                if not is_batch:

                    def _emit_debug_artifacts():
                        try:
                            if ENABLE_SL:
                                history_dir = os.path.join(
                                    result_dir, "sl_history_frames"
                                )
                                if os.path.exists(history_dir):
                                    shutil.rmtree(history_dir)
                                os.makedirs(history_dir, exist_ok=True)

                                if "history_paths" not in best_match:
                                    best_match_abs_path = os.path.join(
                                        app_args.dataset_path, best_match["rel_path"]
                                    )
                                    h_paths, f_paths = get_database_spatial_sequence(
                                        best_match_abs_path,
                                        pose_manager,
                                        SPATIAL_STEPS,
                                    )
                                    best_match["history_paths"] = h_paths
                                    best_match["future_paths"] = f_paths

                                bm_cropped_seq = self.cropper.process_sequence(
                                    os.path.join(
                                        app_args.dataset_path, best_match["rel_path"]
                                    ),
                                    best_match["history_paths"],
                                    best_match.get("future_paths", []),
                                    save_dir=history_dir,
                                )

                            for i, cand in enumerate(candidates):
                                error_val = cand.get("error")
                                error_str = (
                                    f"_err{error_val:.2f}m"
                                    if error_val is not None
                                    else "_errN-A"
                                )
                                dst_name = f"rank{i+1}_sim{cand['vpr_sim']:.3f}{error_str}_{os.path.splitext(cand['basename'])[0]}.jpg"
                                try:
                                    generate_topk_pair(
                                        query_full_path,
                                        cand,
                                        topk_dir,
                                        app_args.dataset_path,
                                        query_pose_info,
                                        dst_name,
                                    )
                                except:
                                    pass

                            try:
                                db_path = generate_debug_dashboard(
                                    query_full_path,
                                    candidates[0],
                                    best_match,
                                    result_dir,
                                    app_args.dataset_path,
                                    query_pose_info,
                                )
                                map_path = generate_trajectory_map(
                                    app_args.dataset_path,
                                    query_pose_info,
                                    candidates[0],
                                    best_match,
                                    query_name,
                                    pose_manager,
                                )
                            except Exception as e:
                                vpr_logger.warning(
                                    f"[Debug] Visualizations generation failed: {e}"
                                )

                            try:
                                _raw1 = candidates[0]
                                shutil.copy2(
                                    query_full_path,
                                    os.path.join(
                                        result_dir,
                                        "query" + os.path.splitext(query_full_path)[1],
                                    ),
                                )
                                for _tag, _c in (
                                    ("rank1_raw", _raw1),
                                    ("final", best_match),
                                ):
                                    _src = os.path.join(
                                        app_args.dataset_path, _c["rel_path"]
                                    )
                                    if os.path.isfile(_src):
                                        shutil.copy2(
                                            _src,
                                            os.path.join(
                                                result_dir,
                                                f"{_tag}_{_c['basename']}",
                                            ),
                                        )
                                with open(
                                    os.path.join(result_dir, "result.json"),
                                    "w",
                                    encoding="utf-8",
                                ) as _fh:
                                    json.dump(
                                        {
                                            "query": query_name,
                                            "query_image": query_full_path,
                                            "rerank_applied": best_match is not _raw1
                                            or getattr(app_args, "rerank", True),
                                            "changed_top1": best_match["rel_path"]
                                            != _raw1["rel_path"],
                                            "zoom": getattr(app_args, "zoom", None),
                                            "final": {
                                                "rank_before": best_match["rank"],
                                                "basename": best_match["basename"],
                                                "vpr_sim": best_match["vpr_sim"],
                                                "error_m": best_match.get("error"),
                                                "pose": best_match.get("pose"),
                                            },
                                            "raw_top1": {
                                                "basename": _raw1["basename"],
                                                "vpr_sim": _raw1["vpr_sim"],
                                                "error_m": _raw1.get("error"),
                                            },
                                            "candidates": [
                                                {
                                                    "rank": c["rank"],
                                                    "basename": c["basename"],
                                                    "vpr_sim": c["vpr_sim"],
                                                    "error_m": c.get("error"),
                                                }
                                                for c in candidates
                                            ],
                                        },
                                        _fh,
                                        ensure_ascii=False,
                                        indent=2,
                                    )
                            except Exception as _exc:
                                vpr_logger.warning(
                                    f"[Debug] 逐 query 证据落盘失败: {_exc}"
                                )

                        except Exception as _exc:
                            vpr_logger.warning(f"[Debug] 后台产物失败: {_exc}")

                    threading.Thread(target=_emit_debug_artifacts, daemon=True).start()

                result_data = {
                    "best_match_image": os.path.join(
                        app_args.dataset_path, best_match["rel_path"]
                    ),
                    "query_image": query_full_path,
                    "debug_dashboard": db_path,
                    "score": best_match["final_score"],
                    "info_payload": best_match["basename"],
                    "pose": best_match["pose"],
                    "query_pose": query_pose,
                    "distance_error": distance_error,
                    "rank": best_match["rank"],
                    "raw_vpr_errors": [c.get("error") for c in candidates],
                }

                dist_str = (
                    f"{distance_error:.3f}m" if distance_error is not None else "N/A"
                )
                log_prefix = "[VPR MATCH]" if not is_batch else "[BATCH MATCH]"
                if (
                    distance_error is not None
                    and distance_error < SUCCESS_THRESHOLD_TEST
                ):
                    dist_str = f"\033[1;32m{dist_str} (hit)\033[0m"
                elif distance_error is not None:
                    dist_str = f"\033[1;31m{dist_str} (false positive)\033[0m"

                vpr_logger.info(
                    f"{log_prefix} Query: {query_name: <{SUCCESS_THRESHOLD_TEST}} | Final: {best_match['basename']: <25} | Physical error: {dist_str}"
                )

                if not is_batch:
                    vpr_logger.info(
                        f"🏁 [END QUERY] Finished query frame {query_name}\n"
                    )

                result_data["inference_time_sec"] = (
                    time.perf_counter() - inference_start
                )
                return result_data

            except Exception as e:
                vpr_logger.error(f"[VPR ERROR] {e}")
                return None
