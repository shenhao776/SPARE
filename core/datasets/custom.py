import os
import math
from decimal import Decimal

from core.datasets.base import PoseAdapter


class CustomAdapter(PoseAdapter):
    pose_format = "custom"

    def load_poses(self, sequence_dir):
        pose_file = os.path.join(sequence_dir, "pose.txt")
        if not os.path.exists(pose_file):
            return {}
        pose_dict = {}
        try:
            with open(pose_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("#"):
                        continue
                    if not line or line.startswith("timestamp"):
                        continue
                    parts = line.split()
                    if len(parts) >= 8:
                        ts_str = parts[0]
                        if "." in ts_str:
                            ns_key = int(Decimal(ts_str) * Decimal("1e9"))
                        else:
                            ns_key = int(ts_str)
                        pose_dict[ns_key] = [float(p) for p in parts[1:8]]
            return pose_dict
        except Exception:
            return {}

    def to_global(self, local_pose, sequence_dir=None):
        return local_pose

    def physical_distance(self, global_a, global_b, sequence_dir=None):
        dx = global_a[0] - global_b[0]
        dy = global_a[1] - global_b[1]
        return math.sqrt(dx * dx + dy * dy)


class CustomWGS84Adapter(CustomAdapter):
    pose_format = "custom_wgs84"

    def physical_distance(self, global_a, global_b, sequence_dir=None):
        lon1, lat1 = global_a[0], global_a[1]
        lon2, lat2 = global_b[0], global_b[1]
        R = 6378137.0
        lat_rad = math.radians(lat1)
        dx_m = math.radians(lon2 - lon1) * R * math.cos(lat_rad)
        dy_m = math.radians(lat2 - lat1) * R
        dz_m = global_a[2] - global_b[2]
        return math.sqrt(dx_m * dx_m + dy_m * dy_m + dz_m * dz_m)
