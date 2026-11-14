import math

import numpy as np


class PoseAdapter:
    pose_format = None

    def load_poses(self, sequence_dir):

        raise NotImplementedError

    def to_global(self, local_pose, sequence_dir=None):

        raise NotImplementedError

    def physical_distance(self, global_a, global_b, sequence_dir=None):

        raise NotImplementedError

    def angular_distance(self, global_a, global_b):

        if not global_a or not global_b:
            return None
        try:
            if len(global_a) >= 7 and len(global_b) >= 7:
                q_a = np.array(global_a[3:7])
                q_b = np.array(global_b[3:7])
                dot_product = np.clip(np.abs(np.dot(q_a, q_b)), -1.0, 1.0)
                angle_rad = 2 * np.arccos(dot_product)
                return math.degrees(angle_rad)
            elif len(global_a) >= 4 and len(global_b) >= 4:
                yaw_a = global_a[3]
                yaw_b = global_b[3]
                diff_rad = abs(yaw_a - yaw_b)
                diff_rad = diff_rad % (2 * math.pi)
                if diff_rad > math.pi:
                    diff_rad = 2 * math.pi - diff_rad
                return math.degrees(diff_rad)
            else:
                return 0.0
        except Exception:
            return 0.0
