import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from core.config import (
    ABLATION_USE_VPRPATCH,
    ABLATION_USE_VPRPATCHSHUF,
    INLIER_NORM_FACTOR,
    LOG1P,
    SPATIAL_STEPS,
    TOP_K,
    SUCCESS_THRESHOLD_TRAIN,
)


def build_sequence_features(
    inliers,
    scale_ratios,
    max_steps,
    use_derivatives=True,
    use_progress=True,
    extra_channels=None,
    use_core=True,
    use_scale=True,
):

    cand_feat = []
    seq_len = min(len(inliers), max_steps)

    for t in range(seq_len):
        if LOG1P:
            cur_inl = math.log1p(inliers[t]) / 4.0
        else:
            cur_inl = inliers[t] / INLIER_NORM_FACTOR

        cur_scale = max(0.0, min(scale_ratios[t], 5.0))

        feat_vector = (
            ([cur_inl, cur_scale] if use_scale else [cur_inl]) if use_core else []
        )

        if use_derivatives:
            d1_sc = (
                (cur_scale - max(0.0, min(scale_ratios[t - 1], 5.0))) if t > 0 else 0.0
            )
            if t > 1:
                prev_d1_sc = max(0.0, min(scale_ratios[t - 1], 5.0)) - max(
                    0.0, min(scale_ratios[t - 2], 5.0)
                )
                d2_sc = d1_sc - prev_d1_sc
            else:
                d2_sc = 0.0
            feat_vector.extend([d1_sc, d2_sc])

        if use_progress:
            progress = t / float(max_steps - 1) if max_steps > 1 else 0.0
            feat_vector.append(progress)

        if extra_channels:
            feat_vector.extend(
                float(ch[t]) if t < len(ch) else 0.0 for ch in extra_channels
            )

        cand_feat.append(feat_vector)

    n_extra = len(extra_channels) if extra_channels else 0
    while len(cand_feat) < max_steps:
        pad_vector = [0.0, 0.0] if use_core else []
        if use_derivatives:
            pad_vector.extend([0.0, 0.0])
        if use_progress:
            progress = len(cand_feat) / float(max_steps - 1) if max_steps > 1 else 0.0
            pad_vector.append(progress)
        pad_vector.extend([0.0] * n_extra)
        cand_feat.append(pad_vector)

    return cand_feat


def normalize_block(cands, field, max_steps, required=False):

    flat = [v for c in cands for v in (c.get(field) or [])]
    if not flat:
        if required:
            raise ValueError(
                f"channel '{field}' was requested but no candidate carries it; "
                f"the state cache predates it (see tools/add_dino_channel.py)"
            )
        return [[0.0] * max_steps for _ in cands]
    lo, hi = min(flat), max(flat)
    rng = (hi - lo) or 1.0
    out = []
    for c in cands:
        seq = [(v - lo) / rng for v in (c.get(field) or [])[:max_steps]]
        out.append(seq + [0.0] * (max_steps - len(seq)))
    return out


SINGLE_FRAME_ARCHS = ("single_frame", "gru_single", "single_frame_linear")


def is_single_frame(arch):
    return str(arch).lower() in SINGLE_FRAME_ARCHS


def head_steps(arch, max_steps):

    return 1 if is_single_frame(arch) else max_steps


def center_frame_view(cands, max_steps):

    centre = max_steps // 2
    out = []
    for c in cands:
        view = dict(c)
        for key, value in c.items():
            if key == "global_pose" or not isinstance(value, list):
                continue
            if key == "dino_sim":
                if value:
                    view[key] = [value[len(value) // 2]]
            elif len(value) == max_steps:
                view[key] = [value[centre]]
        out.append(view)
    return out


WINDOW_MLP_ARCHS = ("mlp_flat", "mlp_pool")


def _require_single_step(steps):
    if steps != 1:
        raise ValueError(
            f"single_frame head expects one step per candidate, got {steps}: "
            "collapse the window with center_frame_view before the forward pass"
        )


def build_extra_channels(
    cands,
    max_steps,
    use_sim,
    use_area,
    use_rank=False,
    use_dino=False,
    use_pose=False,
    use_hull=False,
    use_homog=False,
    use_walk=False,
    use_shape=False,
    use_zoom=False,
    use_zoomshuf=False,
    use_vprpatch=None,
    use_vprpatchshuf=None,
):

    if use_vprpatch is None:
        use_vprpatch = ABLATION_USE_VPRPATCH
    if use_vprpatchshuf is None:
        use_vprpatchshuf = ABLATION_USE_VPRPATCHSHUF
    chans = []
    if use_sim:
        chans.append(normalize_block(cands, "scores", max_steps))
    if use_area:
        chans.append(normalize_block(cands, "area_query", max_steps))
        chans.append(normalize_block(cands, "area_step", max_steps))
    if use_rank:
        span = float(max(len(cands) - 1, 1))
        chans.append([[1.0 - i / span] * max_steps for i in range(len(cands))])
    if use_dino:
        chans.append(normalize_block(cands, "dino_sim", max_steps, required=True))
    if use_pose:
        chans.append(normalize_block(cands, "step_offset_m", max_steps, required=True))
        chans.append(normalize_block(cands, "step_turn_deg", max_steps, required=True))
    if use_hull:
        for key in ("hull_cx", "hull_cy", "hull_aspect", "hull_fill"):
            chans.append(normalize_block(cands, key, max_steps, required=True))
    if use_homog:
        for key in ("H_tx", "H_ty", "H_logdet", "H_persp"):
            chans.append(normalize_block(cands, key, max_steps, required=True))
    if use_walk:
        chans.append(normalize_block(cands, "step_change", max_steps, required=True))
        chans.append(normalize_block(cands, "step_valid", max_steps, required=True))
    if use_shape:
        chans.append(normalize_block(cands, "peak_off", max_steps, required=True))
        chans.append(normalize_block(cands, "peak_sharp", max_steps, required=True))
    if use_zoom:
        for c in cands:
            if "zoom" not in c:
                raise KeyError(
                    "zoom channel enabled but the cache has no "
                    "'zoom' field -- run tools/inject_zoom_channel.py"
                )
        chans.append(
            [
                [
                    math.log2(max(float(z), 1e-6)) - 1.0
                    for z in (
                        c["zoom"][:max_steps]
                        + [c["zoom"][-1]] * max(0, max_steps - len(c["zoom"]))
                    )
                ]
                for c in cands
            ]
        )
    if use_zoomshuf:
        chans.append(
            [
                [
                    math.log2(max(float(z), 1e-6)) - 1.0
                    for z in (
                        c["zoom_shuf"][:max_steps]
                        + [c["zoom_shuf"][-1]] * max(0, max_steps - len(c["zoom_shuf"]))
                    )
                ]
                for c in cands
            ]
        )
    if use_vprpatch or use_vprpatchshuf:
        if max_steps != 1:
            raise ValueError(
                "vprpatch is a single-frame channel (centre frame only); "
                f"got a {max_steps}-step head"
            )
    if use_vprpatch:
        chans.append(normalize_block(cands, "vprp_sim", max_steps, required=True))
    if use_vprpatchshuf:
        chans.append(normalize_block(cands, "vprp_shuf", max_steps, required=True))
    return [[ch[i] for ch in chans] for i in range(len(cands))] if chans else None


class VPRSequenceDataset(Dataset):
    def __init__(
        self,
        data_list,
        spatial_steps,
        use_derivatives=True,
        use_progress=True,
        use_sim=False,
        use_area=False,
        use_rank=False,
        use_dino=False,
        use_pose=False,
        use_hull=False,
        use_homog=False,
        use_walk=False,
        use_shape=False,
        use_zoom=False,
        use_zoomshuf=False,
        arch="gru",
        use_core=True,
        use_scale=True,
        use_vprpatch=None,
        use_vprpatchshuf=None,
    ):
        self.max_steps = sum(spatial_steps) + 1
        self.use_core = use_core
        self.use_scale = use_scale
        self.head_steps = head_steps(arch, self.max_steps)
        self.use_derivatives = use_derivatives
        self.use_progress = use_progress
        self.use_sim = use_sim
        self.use_area = use_area
        self.use_rank = use_rank
        self.use_dino = use_dino
        self.use_pose = use_pose
        self.use_hull = use_hull
        self.use_homog = use_homog
        self.use_walk = use_walk
        self.use_shape = use_shape
        self.use_zoom = use_zoom
        self.use_zoomshuf = use_zoomshuf
        self.use_vprpatch = (
            ABLATION_USE_VPRPATCH if use_vprpatch is None else use_vprpatch
        )
        self.use_vprpatchshuf = (
            ABLATION_USE_VPRPATCHSHUF if use_vprpatchshuf is None else use_vprpatchshuf
        )

        grouped_data = {}
        for d in data_list:
            q_name = d["query"]
            if q_name not in grouped_data:
                grouped_data[q_name] = []
            grouped_data[q_name].append(d)

        self.total_original_queries = len(grouped_data)
        self.features, self.targets, self.masks, self.query_names = [], [], [], []

        dummy_cand_feature = build_sequence_features(
            [],
            [],
            self.head_steps,
            use_derivatives,
            use_progress,
            use_core=use_core,
            use_scale=use_scale,
        )

        for q_name, cands in grouped_data.items():
            cands = sorted(cands, key=lambda x: x.get("rank", 999))[:TOP_K]
            if len(cands) == 0:
                continue

            valid_errors = [
                c.get("error", 999.0) if c.get("error", 999.0) >= 0 else 999.0
                for c in cands
            ]
            target_probs = [0.0] * TOP_K
            weight_sum = 0.0

            sigma = SUCCESS_THRESHOLD_TRAIN / 3.0
            for i, err in enumerate(valid_errors):
                if err <= SUCCESS_THRESHOLD_TRAIN:
                    weight = math.exp(-(err**2) / (2 * sigma**2))
                    target_probs[i] = weight
                    weight_sum += weight

            if weight_sum == 0:
                continue
            target_probs = [p / weight_sum for p in target_probs]

            valid_len = len(cands)
            group_mask = [1.0] * valid_len + [0.0] * (TOP_K - valid_len)

            if self.head_steps != self.max_steps:
                cands = center_frame_view(cands, self.max_steps)
            extra = build_extra_channels(
                cands,
                self.head_steps,
                self.use_sim,
                self.use_area,
                self.use_rank,
                self.use_dino,
                self.use_pose,
                self.use_hull,
                self.use_homog,
                self.use_walk,
                self.use_shape,
                self.use_zoom,
                self.use_zoomshuf,
                use_vprpatch=self.use_vprpatch,
                use_vprpatchshuf=self.use_vprpatchshuf,
            )

            group_feature = []
            for ci, cand in enumerate(cands):
                inliers = cand.get("inliers", [])
                scale_ratios = cand.get("scale_ratio", [])
                seq_feature = build_sequence_features(
                    inliers,
                    scale_ratios,
                    self.head_steps,
                    self.use_derivatives,
                    self.use_progress,
                    extra[ci] if extra else None,
                    use_core=self.use_core,
                    use_scale=self.use_scale,
                )
                group_feature.append(seq_feature)

            n_extra = len(extra[0]) if extra else 0
            pad_seq = [row + [0.0] * n_extra for row in dummy_cand_feature]
            while len(group_feature) < TOP_K:
                group_feature.append(pad_seq)

            self.features.append(group_feature)
            self.targets.append(target_probs)
            self.masks.append(group_mask)
            self.query_names.append(q_name)

        self.features = torch.tensor(self.features, dtype=torch.float32)
        self.targets = torch.tensor(self.targets, dtype=torch.float32)
        self.masks = torch.tensor(self.masks, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (
            self.features[idx],
            self.targets[idx],
            self.masks[idx],
            self.query_names[idx],
        )


class VPRReranking(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, arch="gru", input_norm=True):
        super(VPRReranking, self).__init__()
        self.hidden_dim = hidden_dim
        self.arch = arch.lower()
        self.norm = nn.LayerNorm(input_dim) if input_norm else nn.Identity()

        if self.arch == "gru":
            self.gru = nn.GRU(
                input_dim,
                hidden_dim,
                num_layers=2,
                batch_first=True,
                bidirectional=True,
                dropout=0.2,
            )
            classifier_input_dim = hidden_dim * 2
        elif self.arch == "mlp":
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim * 2),
            )
            classifier_input_dim = hidden_dim * 2
        elif self.arch == "gru_single":
            self.gru = nn.GRU(
                input_dim,
                hidden_dim,
                num_layers=2,
                batch_first=True,
                bidirectional=True,
                dropout=0.2,
            )
            classifier_input_dim = hidden_dim * 2
        elif self.arch == "single_frame_linear":
            self.mlp = nn.Identity()
            classifier_input_dim = input_dim
        elif self.arch in SINGLE_FRAME_ARCHS:
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            classifier_input_dim = hidden_dim
        elif self.arch == "mlp_flat":
            self.window_steps = sum(SPATIAL_STEPS) + 1
            self.mlp = nn.Sequential(
                nn.Linear(input_dim * self.window_steps, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            classifier_input_dim = hidden_dim
        elif self.arch == "mlp_pool":
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim * 2),
            )
            classifier_input_dim = hidden_dim * 2
        elif self.arch == "transformer":
            self.in_proj = nn.Linear(input_dim, hidden_dim)
            self.pos_emb = nn.Parameter(torch.zeros(1, 64, hidden_dim))
            enc_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=hidden_dim * 2,
                dropout=0.2,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
            self.out_proj = nn.Linear(hidden_dim, hidden_dim * 2)
            classifier_input_dim = hidden_dim * 2
        else:
            raise ValueError(f"Unknown architecture: {self.arch}")

        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
        )
        if self.arch == "single_frame_linear":
            self.dropout = nn.Identity()
        if self.arch in ("single_frame", "mlp_flat", "single_frame_linear"):
            self.classifier = nn.Linear(classifier_input_dim, 1)

    def _flat_trunk(self, x_flat):

        n, s, _ = x_flat.shape
        if s < self.window_steps:
            x_flat = nn.functional.pad(x_flat, (0, 0, 0, self.window_steps - s))
        elif s > self.window_steps:
            x_flat = x_flat[:, : self.window_steps]
        return self.mlp(x_flat.reshape(n, -1)).unsqueeze(1)

    def forward(self, x, mask=None):
        if x.dim() == 4:
            B, C, S, F = x.size()
            x_flat = x.view(B * C, S, F)
            x_flat = self.norm(x_flat)

            if self.arch == "gru":
                out, _ = self.gru(x_flat)
            elif self.arch == "mlp":
                out = self.mlp(x_flat)
            elif self.arch == "gru_single":
                _require_single_step(S)
                out, _ = self.gru(x_flat)
            elif self.arch in SINGLE_FRAME_ARCHS:
                _require_single_step(S)
                out = self.mlp(x_flat)
            elif self.arch == "mlp_flat":
                out = self._flat_trunk(x_flat)
            elif self.arch == "mlp_pool":
                out = self.mlp(x_flat)
            elif self.arch == "transformer":
                h = self.in_proj(x_flat) + self.pos_emb[:, : x_flat.size(1)]
                out = self.out_proj(self.encoder(h))

            out = self.dropout(out)
            logits = self.classifier(out)
            if self.arch == "mlp_pool":
                logits = logits.mean(dim=1, keepdim=True)
            S_out = logits.size(1)
            logits = logits.view(B, C, S_out)

            if mask is not None:
                mask_expanded = mask.unsqueeze(-1).expand(-1, -1, S_out)
                logits = logits.masked_fill(mask_expanded == 0, -1e9)

            return logits
        else:
            x = self.norm(x)
            if self.arch == "gru":
                out, _ = self.gru(x)
            elif self.arch == "mlp":
                out = self.mlp(x)
            elif self.arch == "gru_single":
                _require_single_step(x.size(-2))
                out, _ = self.gru(x)
            elif self.arch in SINGLE_FRAME_ARCHS:
                _require_single_step(x.size(-2))
                out = self.mlp(x)
            elif self.arch == "mlp_flat":
                out = self._flat_trunk(x)
            elif self.arch == "mlp_pool":
                out = self.mlp(x)
            elif self.arch == "transformer":
                h = self.in_proj(x) + self.pos_emb[:, : x.size(1)]
                out = self.out_proj(self.encoder(h))
            out = self.dropout(out)
            logits = self.classifier(out)
            if self.arch == "mlp_pool":
                logits = logits.mean(dim=1, keepdim=True)
            return logits

    def predict_softmax(self, x, mask=None):
        logits = self.forward(x, mask)
        return torch.softmax(logits, dim=1)
