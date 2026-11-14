import os

import cv2
import numpy as np
import torch

from core.config import (
    ABLATION_ARCH,
    ABLATION_USE_AREA,
    ABLATION_USE_CORE,
    ABLATION_USE_SCALE,
    ABLATION_USE_DERIVATIVES,
    ABLATION_USE_DINO,
    ABLATION_USE_HOMOG,
    ABLATION_USE_HULL,
    ABLATION_USE_POSE,
    ABLATION_USE_PROGRESS,
    ABLATION_USE_RANK,
    ABLATION_USE_SHAPE,
    ABLATION_USE_SIM,
    ABLATION_USE_WALK,
    ABLATION_USE_ZOOM,
    ABLATION_USE_VPRPATCH,
    ABLATION_USE_VPRPATCHSHUF,
    ABLATION_USE_ZOOMSHUF,
    DINO_K,
    SPATIAL_STEPS,
    TARGET_SCENE,
    TOP_K,
)
from core.vpr_core import (
    GEOMETRY_KEYS,
    get_database_spatial_sequence,
    pose_manager,
)
from sl_agent.vpr_classifier import (
    build_extra_channels,
    build_sequence_features,
    center_frame_view,
    head_steps,
)

_DINO_IMPORTS = None


def _dino_module():

    global _DINO_IMPORTS
    if _DINO_IMPORTS is None:
        import importlib.util

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        spec = importlib.util.spec_from_file_location(
            "_star_add_dino_channel", os.path.join(root, "tools", "add_dino_channel.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _DINO_IMPORTS = mod
    return _DINO_IMPORTS


_VPRP_IMPORTS = None


def _vprp_module():

    global _VPRP_IMPORTS
    if _VPRP_IMPORTS is None:
        import importlib.util

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        spec = importlib.util.spec_from_file_location(
            "_star_add_vprpatch_channel",
            os.path.join(root, "tools", "add_vprpatch_channel.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _VPRP_IMPORTS = mod
    return _VPRP_IMPORTS


_SHAPE_FN = None


def _shape_channels(seq, length):

    global _SHAPE_FN
    if _SHAPE_FN is None:
        import importlib.util

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        spec = importlib.util.spec_from_file_location(
            "_star_inject_step_meta",
            os.path.join(root, "tools", "inject_step_meta.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _SHAPE_FN = mod._shape
    return _SHAPE_FN(seq, length)


class OnlineStateBuilder:

    def __init__(self, service, dataset_path, scene=None, match_batch=1):
        self.service = service
        self.cropper = service.cropper
        self.device = service.args.device
        self.dataset_path = dataset_path
        self.scene = scene or TARGET_SCENE
        self.max_steps = sum(SPATIAL_STEPS) + 1
        self.match_batch = int(match_batch)
        self._pt_cache = {}
        self._win_cache = {}
        self._dino_model = None
        self._db_patch_cache = {}
        self._vprp_model = None
        self._vprp_db_cache = {}

    def _window(self, db_abs_path):

        if db_abs_path in self._win_cache:
            return self._win_cache[db_abs_path]
        h_paths, f_paths, meta = get_database_spatial_sequence(
            db_abs_path, pose_manager, SPATIAL_STEPS, return_meta=True
        )
        abs_seq = list(reversed(h_paths)) + [db_abs_path] + list(f_paths)
        rel_seq = [os.path.relpath(p, self.dataset_path) for p in abs_seq]
        out = (abs_seq, rel_seq, meta)
        self._win_cache[db_abs_path] = out
        return out

    def _step_features(self, rel_path):

        if rel_path in self._pt_cache:
            return self._pt_cache[rel_path]
        asset = os.path.join("assets", "sequence_assets", self.scene, rel_path + ".pt")
        if not os.path.exists(asset):
            raise FileNotFoundError(
                f"sequence asset missing: {asset}. The map's per-frame SuperPoint "
                "features are a deployment prerequisite -- build them with "
                "`python reproduce.py` for this scene before running online."
            )
        data = torch.load(asset, map_location="cpu", weights_only=False)
        lg = data.get("lightfeats", data.get("lightglue_feats", []))
        paths = data.get("rel_seq_paths", [])
        feats = None
        if lg:
            idx = len(paths) // 2 if paths else 0
            feats = lg[min(idx, len(lg) - 1)]
        if len(self._pt_cache) >= 512:
            del self._pt_cache[next(iter(self._pt_cache))]
        self._pt_cache[rel_path] = feats
        return feats

    def _dino_patches(self, paths, is_db):

        mod = _dino_module()
        if self._dino_model is None:
            self._dino_model = mod.load_model()
        mean, std = mod.MEAN.to(mod.DEV), mod.STD.to(mod.DEV)
        out = []
        todo = []
        for p in paths:
            if is_db and p in self._db_patch_cache:
                out.append(self._db_patch_cache[p])
            else:
                out.append(None)
                todo.append(p)
        if todo:
            tensors, keep = [], []
            for p in todo:
                im = cv2.imread(p)
                if im is None:
                    continue
                im = cv2.cvtColor(
                    cv2.resize(im, (mod.IMG_W, mod.IMG_H)), cv2.COLOR_BGR2RGB
                )
                tensors.append(torch.from_numpy(im).float().permute(2, 0, 1))
                keep.append(p)
            if tensors:
                t = (torch.stack(tensors).to(mod.DEV) / 255.0 - mean) / std
                with torch.no_grad():
                    feats = self._dino_model.forward_features(t)["x_norm_patchtokens"]
                feats = torch.nn.functional.normalize(feats, dim=-1)
                feats = feats.half().cpu().numpy()
                got = {p: feats[i] for i, p in enumerate(keep)}
                for i, p in enumerate(paths):
                    if out[i] is None and p in got:
                        out[i] = got[p]
                        if is_db:
                            if len(self._db_patch_cache) >= 1024:
                                del self._db_patch_cache[
                                    next(iter(self._db_patch_cache))
                                ]
                            self._db_patch_cache[p] = got[p]
        return out

    def build_rows(
        self, query_cv, query_path, candidates, db_lookup, q_desc, zoom=None
    ):

        query_feats = self.cropper.extract_image(query_cv)
        img_size = None
        size_t = (query_feats or {}).get("image_size")
        if size_t is not None:
            try:
                img_size = tuple(float(v) for v in size_t[0].tolist())
            except Exception:
                img_size = None

        q_desc = np.asarray(q_desc, dtype=np.float32).reshape(-1)
        q_norm = float(np.linalg.norm(q_desc)) or 1.0

        need_vprp = ABLATION_USE_VPRPATCH or ABLATION_USE_VPRPATCHSHUF
        q_vprp = self._vprp_tokens(query_path, is_db=False) if need_vprp else None

        need_dino = ABLATION_USE_DINO
        q_patch = None
        if need_dino:
            q_patch = self._dino_patches([query_path], is_db=False)[0]
            if q_patch is not None:
                q_patch = np.asarray(q_patch, dtype=np.float32)

        single_frame = head_steps(ABLATION_ARCH, self.max_steps) == 1

        rows = []
        for cand in candidates:
            db_abs = os.path.join(self.dataset_path, cand["rel_path"])
            if single_frame:
                abs_seq = [db_abs]
                rel_seq = [cand["rel_path"]]
                meta = [{"change": 0.0, "dist": 0.0, "valid": True}]
            else:
                abs_seq, rel_seq, meta = self._window(db_abs)

            inliers, area_step, area_query, scale_ratio = [], [], [], []
            geo_seq = {k: [] for k in GEOMETRY_KEYS}
            scores = []

            step_gpu_seq = [
                (self.cropper.unpack_offline_features(sf, self.device) if sf else None)
                for sf in (self._step_features(rel_p) for rel_p in rel_seq)
            ]
            if self.match_batch > 1:
                matched = self.cropper.match_features_batch(
                    query_feats, step_gpu_seq, max_batch=self.match_batch
                )
            else:
                matched = [
                    (
                        self.cropper.match_features(query_feats, sg)
                        if sg
                        else ([], [], [])
                    )
                    for sg in step_gpu_seq
                ]

            for step_idx, rel_p in enumerate(rel_seq):
                vi, sr, kas, kaq = 0, 0.0, 0.0, 0.0
                geo = dict.fromkeys(GEOMETRY_KEYS, 0.0)
                if step_gpu_seq[step_idx] is not None:
                    kp_q, kp_s, matches = matched[step_idx]
                    vi, sr, kas, kaq, geo = self.cropper.compute_geometric_metrics(
                        kp_q, kp_s, matches, with_geometry=True, img_size=img_size
                    )
                inliers.append(vi)
                area_step.append(round(kas, 2))
                area_query.append(round(kaq, 2))
                scale_ratio.append(round(sr, 4))
                for gk in GEOMETRY_KEYS:
                    geo_seq[gk].append(round(float(geo[gk]), 5))

                sp_feat = db_lookup.get(rel_p)
                if sp_feat is not None:
                    cos = float(
                        np.dot(sp_feat, q_desc)
                        / ((np.linalg.norm(sp_feat) or 1.0) * q_norm)
                    )
                    scores.append(round(max(0.0, cos), 4))
                else:
                    scores.append(0.0)

            row = {
                "candidate_name": os.path.basename(cand["rel_path"]),
                "rank": cand.get("rank", len(rows) + 1),
                "inliers": inliers,
                "scale_ratio": scale_ratio,
                "area_step": area_step,
                "area_query": area_query,
                "scores": scores,
                "global_pose": cand.get("global_pose") or [],
            }
            row.update(geo_seq)
            row.update(self._pose_channels(abs_seq, cand))
            row["peak_off"], row["peak_sharp"] = _shape_channels(
                inliers, self.max_steps
            )
            row["step_change"] = [float(m.get("change") or 0.0) for m in meta]
            row["step_valid"] = [1.0 if m.get("valid", True) else 0.0 for m in meta]
            if zoom is not None:
                row["zoom"] = [float(zoom)] * len(inliers)
                row["zoom_shuf"] = list(row["zoom"])
            if need_dino:
                row["dino_sim"] = self._dino_row(abs_seq, q_patch)
            if need_vprp:
                d = self._vprp_tokens(abs_seq[len(abs_seq) // 2], is_db=True)
                v = round(_vprp_module().score(q_vprp, d), 5)
                row["vprp_sim"] = [v]
                row["vprp_shuf"] = [v]
            rows.append(row)
        return rows

    def _vprp_tokens(self, path, is_db):

        mod = _vprp_module()
        if self._vprp_model is None:
            self._vprp_model = mod.load_model()
        if is_db and path in self._vprp_db_cache:
            return self._vprp_db_cache[path]
        tok = mod.forward_tokens(self._vprp_model, mod.load_tensor(path)).to(mod.DEV)
        if is_db:
            if len(self._vprp_db_cache) >= 1024:
                self._vprp_db_cache.pop(next(iter(self._vprp_db_cache)))
            self._vprp_db_cache[path] = tok
        return tok

    def _pose_channels(self, abs_seq, cand):

        centre = cand.get("global_pose") or None
        if not centre:
            info = pose_manager.get_pose_info(
                os.path.join(self.dataset_path, cand["rel_path"])
            )
            centre = info["global"] if info and info.get("global") else None
        n = len(abs_seq)
        if centre is None:
            return {"step_offset_m": [0.0] * n, "step_turn_deg": [0.0] * n}
        mid = n // 2
        offs, turns = [], []
        for i, frame in enumerate(abs_seq):
            info = pose_manager.get_pose_info(frame)
            g = info["global"] if info and info.get("global") else None
            sign = -1.0 if i < mid else 1.0
            if g is None:
                offs.append(0.0)
                turns.append(0.0)
                continue
            d = pose_manager.calculate_physical_distance(frame, centre, g)
            t = pose_manager.calculate_angular_distance(centre, g)
            offs.append(sign * float(abs(d or 0.0)))
            turns.append(sign * float(t or 0.0))
        return {"step_offset_m": offs, "step_turn_deg": turns}

    def _dino_row(self, abs_seq, q_patch):

        if q_patch is None:
            return []
        mod = _dino_module()
        win = abs_seq
        if len(win) > DINO_K:
            start = (len(win) - DINO_K) // 2
            win = win[start : start + DINO_K]
        patches = self._dino_patches(win, is_db=True)
        out = []
        for pv in patches:
            if pv is None:
                out.append(0.0)
            else:
                out.append(
                    round(mod.patch_score(q_patch, np.asarray(pv, dtype=np.float32)), 5)
                )
        return out

    def build_state_tensor(self, rows):

        steps = head_steps(ABLATION_ARCH, self.max_steps)
        if steps != self.max_steps:
            rows = center_frame_view(rows, self.max_steps)
        extra = build_extra_channels(
            rows,
            steps,
            ABLATION_USE_SIM,
            ABLATION_USE_AREA,
            ABLATION_USE_RANK,
            ABLATION_USE_DINO,
            ABLATION_USE_POSE,
            ABLATION_USE_HULL,
            ABLATION_USE_HOMOG,
            ABLATION_USE_WALK,
            ABLATION_USE_SHAPE,
            ABLATION_USE_ZOOM,
            ABLATION_USE_ZOOMSHUF,
        )
        n_extra = len(extra[0]) if extra else 0
        feats = []
        for ci, row in enumerate(rows):
            feats.append(
                build_sequence_features(
                    row.get("inliers", []),
                    row.get("scale_ratio", []),
                    steps,
                    use_derivatives=ABLATION_USE_DERIVATIVES,
                    use_progress=ABLATION_USE_PROGRESS,
                    extra_channels=extra[ci] if extra else None,
                    use_core=ABLATION_USE_CORE,
                    use_scale=ABLATION_USE_SCALE,
                )
            )
        dummy = build_sequence_features(
            [],
            [],
            steps,
            use_derivatives=ABLATION_USE_DERIVATIVES,
            use_progress=ABLATION_USE_PROGRESS,
            use_core=ABLATION_USE_CORE,
            use_scale=ABLATION_USE_SCALE,
        )
        pad_steps = [r + [0.0] * n_extra for r in dummy]
        n_valid = len(feats)
        while len(feats) < TOP_K:
            feats.append(pad_steps)
        x = torch.tensor([feats], dtype=torch.float32).to(self.device)
        mask = torch.tensor(
            [[1.0] * n_valid + [0.0] * (TOP_K - n_valid)], dtype=torch.float32
        ).to(self.device)
        return x, mask
