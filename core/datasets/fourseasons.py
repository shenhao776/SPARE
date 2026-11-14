import os
import math

import numpy as np
from scipy.spatial.transform import Rotation as R_sci

from core.datasets.base import PoseAdapter


class FourSeasonsAdapter(PoseAdapter):
    pose_format = "fourseasons"

    def __init__(self):
        self._trans_cache = {}

    def load_poses(self, sequence_dir):
        gnss_file = os.path.join(sequence_dir, "GNSSPoses.txt")
        if not os.path.exists(gnss_file):
            return {}
        try:
            pose_dict = {}
            with open(gnss_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("#") or not line:
                        continue
                    parts = line.replace(",", " ").split()
                    if len(parts) >= 4:
                        ns_key = int(parts[0])
                        pose_dict[ns_key] = [float(p) for p in parts[1:]]
            return pose_dict
        except Exception:
            return {}

    def load_transformations(self, sequence_dir):
        trans_file = os.path.join(sequence_dir, "Transformations.txt")
        if not os.path.exists(trans_file):
            trans_file = os.path.join(
                sequence_dir, "undistorted_images", "Transformations.txt"
            )
        if not os.path.exists(trans_file):
            return None
        trans = {"scale": 1.0}
        try:
            with open(trans_file, "r") as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    if line.startswith("# transform_w_gpsw"):
                        vals = list(map(float, lines[i + 1].strip().split(",")))
                        trans["t_w_gpsw"] = np.array(vals[0:3])
                        trans["q_w_gpsw"] = np.array(vals[3:7])
                    elif line.startswith("# transform_e_gpsw"):
                        vals = list(map(float, lines[i + 1].strip().split(",")))
                        trans["t_e_gpsw"] = np.array(vals[0:3])
                        trans["q_e_gpsw"] = np.array(vals[3:7])
                    elif line.startswith("# GNSS scale"):
                        trans["scale"] = float(lines[i + 1].strip())
            return trans
        except Exception:
            return None

    def _get_trans(self, sequence_dir):
        if sequence_dir not in self._trans_cache:
            self._trans_cache[sequence_dir] = self.load_transformations(sequence_dir)
        return self._trans_cache[sequence_dir]

    def to_global(self, local_pose, sequence_dir=None):
        return self.local_to_global(local_pose, self._get_trans(sequence_dir))

    def local_to_global(self, p_gnss_w, trans):
        if not trans or "t_e_gpsw" not in trans or "q_w_gpsw" not in trans:
            return None
        r_e_gpsw = R_sci.from_quat(trans["q_e_gpsw"])
        t_e_gpsw = trans["t_e_gpsw"]
        r_w_gpsw = R_sci.from_quat(trans["q_w_gpsw"])
        t_w_gpsw = trans["t_w_gpsw"]
        scale = trans["scale"]
        p_w = np.array(p_gnss_w[:3])
        p_gpsw = r_w_gpsw.inv().apply(p_w - t_w_gpsw) * scale
        p_e = r_e_gpsw.apply(p_gpsw) + t_e_gpsw

        yaw = 0.0
        if len(p_gnss_w) >= 7:
            try:
                q_local = p_gnss_w[3:7]
                r_local = R_sci.from_quat(q_local)
                r_global = r_e_gpsw * r_w_gpsw.inv() * r_local
                euler = r_global.as_euler("zyx", degrees=False)
                yaw = euler[0]
            except Exception:
                pass
        res = p_e.tolist()
        res.append(yaw)
        return res

    def physical_distance(self, global_a, global_b, sequence_dir=None):
        try:
            ref_trans = self._get_trans(sequence_dir)
            if ref_trans and "q_e_gpsw" in ref_trans and "t_e_gpsw" in ref_trans:
                ref_r = R_sci.from_quat(ref_trans["q_e_gpsw"]).inv()
                ref_t = ref_trans["t_e_gpsw"]
                local_a = ref_r.apply(np.array(global_a[:3]) - ref_t)
                local_b = ref_r.apply(np.array(global_b[:3]) - ref_t)
                dx = local_a[0] - local_b[0]
                dy = local_a[1] - local_b[1]
                return math.sqrt(dx * dx + dy * dy)
            return math.dist(global_a[:3], global_b[:3])
        except Exception:
            return math.dist(global_a[:3], global_b[:3])
