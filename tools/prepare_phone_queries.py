import argparse
import glob
import json
import os
import shutil
import struct
import sys

import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROTATIONS = {
    "ccw90": cv2.ROTATE_90_COUNTERCLOCKWISE,
    "cw90": cv2.ROTATE_90_CLOCKWISE,
    "180": cv2.ROTATE_180,
    "none": None,
}


def _varint(buf, i):
    result = 0
    shift = 0
    while True:
        byte = buf[i]
        i += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, i
        shift += 7


def _iter_protobuf(buf):

    i = 0
    end = len(buf)
    while i < end:
        key, i = _varint(buf, i)
        field, wire = key >> 3, key & 7
        if wire == 0:
            value, i = _varint(buf, i)
            yield field, wire, value
        elif wire == 2:
            length, i = _varint(buf, i)
            yield field, wire, buf[i : i + length]
            i += length
        elif wire == 5:
            yield field, wire, buf[i : i + 4]
            i += 4
        elif wire == 1:
            yield field, wire, buf[i : i + 8]
            i += 8
        else:
            raise ValueError(f"unsupported protobuf wire type {wire} at byte {i}")


def read_frame_timestamps(meta_path, sync_path):

    with open(sync_path, "r", encoding="utf-8") as handle:
        sync = json.load(handle)
    offset = int(sync["unix_minus_elapsed_realtime_ns"])

    with open(meta_path, "rb") as handle:
        blob = handle.read()

    stamps = []
    for field, wire, payload in _iter_protobuf(blob):
        if field != 5 or wire != 2:
            continue
        elapsed = index = None
        for sub_field, sub_wire, value in _iter_protobuf(payload):
            if sub_wire != 0:
                continue
            if sub_field == 1:
                elapsed = value
            elif sub_field == 2:
                index = value
        if elapsed is not None and index is not None:
            stamps.append((index - 1, elapsed + offset))

    stamps.sort()
    return stamps, sync


def read_poses(bag_paths, topic):

    from rosbags.rosbag2 import Reader
    from rosbags.typesys import Stores, get_typestore

    typestore = get_typestore(Stores.ROS2_HUMBLE)
    rows = []
    skipped = []
    for bag_dir in bag_paths:
        with Reader(bag_dir) as reader:
            conns = [c for c in reader.connections if c.topic == topic]
            if not conns:
                skipped.append(os.path.basename(bag_dir))
                continue
            for conn, _, raw in reader.messages(connections=conns):
                msg = typestore.deserialize_cdr(raw, conn.msgtype)
                stamp = msg.header.stamp
                pos = msg.pose.pose.position
                ori = msg.pose.pose.orientation
                rows.append(
                    (
                        stamp.sec + stamp.nanosec * 1e-9,
                        pos.x,
                        pos.y,
                        pos.z,
                        ori.x,
                        ori.y,
                        ori.z,
                        ori.w,
                        msg.pose.covariance[0],
                    )
                )
    if not rows:
        raise SystemExit(f"no `{topic}` messages found in {len(bag_paths)} bag(s)")
    rows.sort()
    return np.asarray(rows, dtype=np.float64), skipped


def _slerp(qa, qb, ratio):
    dot = float(np.dot(qa, qb))
    if dot < 0.0:
        qb, dot = -qb, -dot
    if dot > 0.9995:
        out = qa + ratio * (qb - qa)
        return out / np.linalg.norm(out)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.sin(theta)
    return (
        np.sin((1 - ratio) * theta) / sin_theta * qa
        + np.sin(ratio * theta) / sin_theta * qb
    )


def interpolate_pose(poses, t, max_dt, max_gap=None):

    times = poses[:, 0]
    idx = int(np.searchsorted(times, t))
    lo, hi = idx - 1, idx
    if lo < 0 or hi >= len(times):
        near = 0 if lo < 0 else len(times) - 1
        if abs(times[near] - t) > max_dt:
            return None
        return poses[near, 1:4].copy(), poses[near, 4:8].copy(), poses[near, 8]

    gap = 2 * max_dt if max_gap is None else max_gap
    if times[hi] - times[lo] > gap:
        near = lo if abs(t - times[lo]) <= abs(times[hi] - t) else hi
        if abs(times[near] - t) > max_dt:
            return None
        return poses[near, 1:4].copy(), poses[near, 4:8].copy(), poses[near, 8]

    span = times[hi] - times[lo]
    ratio = 0.0 if span <= 0 else (t - times[lo]) / span
    xyz = poses[lo, 1:4] + ratio * (poses[hi, 1:4] - poses[lo, 1:4])
    quat = _slerp(poses[lo, 4:8], poses[hi, 4:8], ratio)
    return xyz, quat, max(poses[lo, 8], poses[hi, 8])


def yaw_of(quat):
    x, y, z, w = quat
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def angle_between(yaw_a, yaw_b):
    return abs(np.arctan2(np.sin(yaw_a - yaw_b), np.cos(yaw_a - yaw_b)))


class FrameGrader:

    def __init__(
        self,
        face_min_frac,
        enable_faces=True,
        detect_scale=0.5,
        min_neighbors=6,
        clahe_mode="always",
        clahe_below=80.0,
    ):
        self.face_min_frac = face_min_frac
        self.detect_scale = detect_scale
        self.min_neighbors = min_neighbors
        self.clahe_mode = clahe_mode
        self.clahe_below = clahe_below
        self.cascades = []
        if enable_faces:
            for name in (
                "haarcascade_frontalface_default.xml",
                "haarcascade_profileface.xml",
            ):
                path = os.path.join(cv2.data.haarcascades, name)
                if os.path.exists(path):
                    self.cascades.append(cv2.CascadeClassifier(path))
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    def measure(self, image):

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        smooth = cv2.GaussianBlur(gray, (5, 5), 0)
        gx = cv2.Sobel(smooth, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(smooth, cv2.CV_64F, 0, 1, ksize=3)
        sharpness = float((gx * gx + gy * gy).mean())
        relative = sharpness / (float(gray.var()) + 1e-6)
        return sharpness, float(gray.mean()), gray, relative

    def has_close_face(self, gray, brightness=None):

        if not self.cascades:
            return False
        height, width = gray.shape[:2]
        min_side = max(8, int(self.face_min_frac * height))
        scale = self.detect_scale
        if scale != 1.0:
            gray = cv2.resize(
                gray, (max(1, int(width * scale)), max(1, int(height * scale)))
            )

        use_clahe = self.clahe_mode == "always" or (
            self.clahe_mode == "auto"
            and (brightness is None or brightness < self.clahe_below)
        )
        probe = self.clahe.apply(gray) if use_clahe else gray

        detect_min = max(8, int(min_side * scale))
        for cascade in self.cascades:
            found = cascade.detectMultiScale(
                probe,
                scaleFactor=1.15,
                minNeighbors=self.min_neighbors,
                minSize=(detect_min, detect_min),
            )
            if len(found) > 0:
                return True
        return False


BAG_FILE_EXTS = (".db3", ".mcap")


def find_bags(session):

    roots = sorted(
        {
            os.path.normpath(os.path.dirname(p))
            for p in glob.glob(
                os.path.join(session, "**", "metadata.yaml"), recursive=True
            )
        }
    )
    covered = tuple(root + os.sep for root in roots)
    loose = []
    for ext in BAG_FILE_EXTS:
        for path in glob.glob(os.path.join(session, "**", f"*{ext}"), recursive=True):
            path = os.path.normpath(path)
            if not os.path.isfile(path) or path.startswith(covered):
                continue
            loose.append(path)
    return sorted(set(roots + loose))


def discover_inputs(session, phone_dirs, bag_paths):
    if not phone_dirs:
        phone_dirs = sorted(
            os.path.dirname(p)
            for p in glob.glob(
                os.path.join(session, "**", "video_recording.mp4"), recursive=True
            )
        )
    if not bag_paths:
        bag_paths = find_bags(session)
    if not phone_dirs:
        raise SystemExit(f"no video_recording.mp4 found under {session}")
    if not bag_paths:
        raise SystemExit(
            f"no rosbag2 dir (metadata.yaml) or bare {'/'.join(BAG_FILE_EXTS)} found under {session}"
        )
    return phone_dirs, bag_paths


def build_plan(stamps, poses, args, counts):

    max_dt = args.max_dt_ms / 1000.0
    max_gap = args.max_pose_gap_ms / 1000.0 if args.max_pose_gap_ms else None
    plan = []
    last_eval_t = -np.inf
    for frame_index, stamp_ns in stamps:
        counts["frames"] += 1
        t = stamp_ns * 1e-9
        fix = interpolate_pose(poses, t, max_dt, max_gap)
        if fix is None:
            counts["no_pose"] += 1
            continue
        xyz, quat, cov = fix
        if cov > args.max_cov:
            counts["high_cov"] += 1
            continue
        if t - last_eval_t < args.min_gap_s:
            counts["time_throttled"] += 1
            continue
        last_eval_t = t
        plan.append((frame_index, stamp_ns, xyz, quat, cov))
    return plan


def sample_sharpness(video_path, plan, grader, rotation, budget):

    if not plan or budget <= 0:
        return []
    step = max(1, len(plan) // budget)
    wanted = {plan[i][0] for i in range(0, len(plan), step)}
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise SystemExit(f"cannot open {video_path}")
    scores = []
    index = 0
    highest = max(wanted)
    while index <= highest:
        if not capture.grab():
            break
        if index in wanted:
            ok, frame = capture.retrieve()
            if ok:
                if rotation is not None:
                    frame = cv2.rotate(frame, rotation)
                scores.append(grader.measure(frame)[0])
        index += 1
    capture.release()
    return scores


def process(args):
    phone_dirs, bag_paths = discover_inputs(args.session, args.phone_video, args.bags)
    print(f"[prep] phone videos : {len(phone_dirs)}")
    for d in phone_dirs:
        print(f"         - {d}")
    print(f"[prep] bag inputs   : {len(bag_paths)}")
    for b in bag_paths[:8]:
        print(f"         - {b}")
    if len(bag_paths) > 8:
        print(f"         ... and {len(bag_paths) - 8} more")

    poses, no_topic = read_poses(bag_paths, args.pose_topic)
    if no_topic:
        print(
            f"[prep] warning: {len(no_topic)} bag(s) carry no {args.pose_topic}: {', '.join(no_topic)}"
        )
    covered = float(poses[-1, 0] - poses[0, 0])
    print(
        f"[prep] {len(poses)} poses over {covered:.0f}s span; "
        f"cov p50={np.median(poses[:, 8]):.3f} p95={np.percentile(poses[:, 8], 95):.3f} "
        f"max={poses[:, 8].max():.3f}"
    )

    rotation = ROTATIONS[args.rotate]
    grader = FrameGrader(
        args.face_min_frac,
        enable_faces=not args.no_face_filter,
        detect_scale=args.face_detect_scale,
        min_neighbors=args.face_min_neighbors,
        clahe_mode=args.face_clahe,
        clahe_below=args.face_clahe_below,
    )

    final_out = args.out
    writing = not args.stats_only and not args.dry_run
    if writing:
        existing = os.path.join(final_out, "undistorted_images")
        if os.path.isdir(existing) and os.listdir(existing) and not args.force:
            raise SystemExit(f"{existing} already populated; pass --force to rebuild")
        args.out = final_out + ".partial"
        shutil.rmtree(args.out, ignore_errors=True)
    image_dir = os.path.join(args.out, "undistorted_images")
    if writing:
        os.makedirs(image_dir, exist_ok=True)

    accepted = []
    last_xyz = None
    last_yaw = None
    sharpness_log = []
    counts = {
        "frames": 0,
        "no_pose": 0,
        "high_cov": 0,
        "too_close": 0,
        "time_throttled": 0,
        "decoded": 0,
        "blurry": 0,
        "too_dark": 0,
        "face": 0,
        "accepted": 0,
    }

    planned = []
    for phone_dir in phone_dirs:
        meta_path = os.path.join(phone_dir, "video_meta.pb3")
        sync_path = os.path.join(phone_dir, "time_sync.json")
        video_path = os.path.join(phone_dir, "video_recording.mp4")
        if not (os.path.exists(meta_path) and os.path.exists(sync_path)):
            print(f"[prep] skip {phone_dir}: missing video_meta.pb3 / time_sync.json")
            continue

        stamps, sync = read_frame_timestamps(meta_path, sync_path)
        if not sync.get("ntp_calibration_succeeded", False):
            print(
                f"[prep] warning: {os.path.basename(phone_dir)} has no successful NTP calibration"
            )
        print(
            f"[prep] {os.path.basename(phone_dir)}: {len(stamps)} frames, "
            f"{(stamps[-1][1] - stamps[0][1]) * 1e-9:.0f}s"
        )

        probe = cv2.VideoCapture(video_path)
        video_frames = (
            int(probe.get(cv2.CAP_PROP_FRAME_COUNT)) if probe.isOpened() else -1
        )
        probe.release()
        lo_idx, hi_idx = stamps[0][0], stamps[-1][0]
        print(
            f"[prep]   meta {len(stamps)} records spanning index {lo_idx}..{hi_idx} "
            f"vs video {video_frames} frames"
        )
        tol = args.max_frame_count_mismatch

        if lo_idx != 0:
            raise SystemExit(
                f"{os.path.basename(phone_dir)}: meta frame indices start at {lo_idx}, "
                f"not 0 -- every pose would be offset by {lo_idx} frames"
            )
        if hi_idx + 1 != len(stamps):
            raise SystemExit(
                f"{os.path.basename(phone_dir)}: {len(stamps)} meta records span "
                f"index 0..{hi_idx}; {hi_idx + 1 - len(stamps)} are missing, so frame "
                f"indices past the first gap map to the wrong images"
            )
        if video_frames <= 0:
            if not args.allow_unknown_frame_count:
                raise SystemExit(
                    f"{os.path.basename(phone_dir)}: the container does not report a "
                    f"frame count, so the index mapping cannot be checked; pass "
                    f"--allow-unknown-frame-count to proceed anyway"
                )
            print("[prep]   warning: frame count unavailable; index mapping unchecked")
        elif hi_idx + 1 > video_frames + tol:
            raise SystemExit(
                f"{os.path.basename(phone_dir)}: meta describes {hi_idx + 1} frames but "
                f"the video holds {video_frames} -- indices cannot be trusted"
            )
        elif video_frames - len(stamps) > tol:
            print(
                f"[prep]   warning: {video_frames - len(stamps)} video frames have no "
                f"meta record; they are skipped"
            )

        plan = build_plan(stamps, poses, args, counts)
        if not plan:
            print(f"[prep]   no pose-backed frames in this video")
            continue
        print(f"[prep]   {len(plan)} frames pose-backed and worth grading")
        planned.append((phone_dir, video_path, plan))

    effective_min_sharpness = args.min_sharpness
    if args.min_sharpness_quantile is not None and planned:
        budget = max(1, args.sharpness_sample // len(planned))
        pooled = []
        for _, video_path, plan in planned:
            pooled.extend(sample_sharpness(video_path, plan, grader, rotation, budget))
        MIN_SAMPLE = 60
        if len(pooled) < MIN_SAMPLE:
            raise SystemExit(
                f"--min-sharpness-quantile asked for an adaptive gate but only "
                f"{len(pooled)} frames could be sampled (need {MIN_SAMPLE}). "
                f"Either the plan is nearly empty (check the pose sync first with "
                f"--stats-only) or --sharpness-sample-budget is too low. Re-run "
                f"without --min-sharpness-quantile to use the fixed floor "
                f"{args.min_sharpness:.0f} deliberately."
            )
        q = float(np.percentile(pooled, 100.0 * args.min_sharpness_quantile))
        effective_min_sharpness = max(args.min_sharpness, q)
        print(
            f"[prep] sharpness pre-pass: {len(pooled)} sampled, "
            f"q{args.min_sharpness_quantile:.2f}={q:.0f}, "
            f"floor={args.min_sharpness:.0f} -> gate={effective_min_sharpness:.0f}"
        )

    for phone_dir, video_path, plan in planned:
        plan_by_index = {p[0]: p for p in plan}
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise SystemExit(f"cannot open {video_path}")

        video_index = 0
        while True:
            if not capture.grab():
                break
            entry = plan_by_index.get(video_index)
            video_index += 1
            if entry is None:
                continue
            frame_index, stamp_ns, xyz, quat, cov = entry

            yaw = yaw_of(quat)
            if last_xyz is not None:
                moved = float(np.linalg.norm(xyz[:2] - last_xyz[:2]))
                turned = np.degrees(angle_between(yaw, last_yaw))
                if moved < args.min_dist and turned < args.min_angle:
                    counts["too_close"] += 1
                    continue

            ok, frame = capture.retrieve()
            if not ok:
                continue
            if rotation is not None:
                frame = cv2.rotate(frame, rotation)
            counts["decoded"] += 1

            sharpness, brightness, gray, relative = grader.measure(frame)
            sharpness_log.append((sharpness, brightness, relative))
            if sharpness < effective_min_sharpness:
                counts["blurry"] += 1
                continue
            if brightness < args.min_brightness:
                counts["too_dark"] += 1
                continue
            if grader.has_close_face(gray, brightness):
                counts["face"] += 1
                continue

            counts["accepted"] += 1
            accepted.append((stamp_ns, xyz.copy(), quat.copy()))
            last_xyz, last_yaw = xyz.copy(), yaw
            if not args.stats_only and not args.dry_run:
                cv2.imwrite(
                    os.path.join(image_dir, f"{stamp_ns}.jpg"),
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality],
                )
        capture.release()

    print("\n[prep] frame accounting")
    for key, value in counts.items():
        print(f"         {key:16s} {value}")
    observed = {}
    if sharpness_log:
        arr = np.asarray(sharpness_log)
        qs = [5, 25, 50, 75, 95]
        observed["sharpness"] = {
            f"p{q}": float(np.percentile(arr[:, 0], q)) for q in qs
        }
        observed["brightness"] = {
            f"p{q}": float(np.percentile(arr[:, 1], q)) for q in qs
        }
        observed["sharpness_contrast_normalised"] = {
            f"p{q}": float(np.percentile(arr[:, 2], q)) for q in qs
        }
        observed["effective_min_sharpness"] = float(effective_min_sharpness)
        print(
            "[prep] sharpness percentiles "
            + "  ".join(f"p{q}={np.percentile(arr[:, 0], q):.0f}" for q in qs)
            + f"   (gate={effective_min_sharpness:.0f})"
        )
        print(
            "[prep] brightness percentiles "
            + "  ".join(f"p{q}={np.percentile(arr[:, 1], q):.0f}" for q in qs)
        )
        if args.min_sharpness > 0 and args.min_sharpness_quantile is None:
            print(
                "[prep] note: these percentiles are conditioned on the gate that was "
                "active -- pass --min-sharpness 0 for an unbiased session distribution"
            )

    if len(accepted) > 1:
        ordered = sorted(accepted, key=lambda item: item[0])
        xy = np.asarray([a[1][:2] for a in ordered])
        gaps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        secs = np.diff(np.asarray([a[0] for a in ordered], dtype=np.float64)) * 1e-9
        worst = int(np.argmax(gaps))
        observed["spacing_m"] = {
            "p50": float(np.percentile(gaps, 50)),
            "p95": float(np.percentile(gaps, 95)),
            "max": float(gaps.max()),
            "max_at_xy": [float(xy[worst, 0]), float(xy[worst, 1])],
        }
        observed["spacing_s"] = {
            "p50": float(np.percentile(secs, 50)),
            "p95": float(np.percentile(secs, 95)),
            "max": float(secs.max()),
        }
        print(
            f"[prep] accepted spacing: p50={np.percentile(gaps, 50):.2f}m "
            f"p95={np.percentile(gaps, 95):.2f}m max={gaps.max():.2f}m "
            f"at (x={xy[worst, 0]:.1f}, y={xy[worst, 1]:.1f})"
        )

    if args.stats_only:
        print("[prep] --stats-only: nothing written")
        return
    if args.dry_run:
        print(f"[prep] --dry-run: would write {len(accepted)} frames to {args.out}")
        return
    if not accepted:
        raise SystemExit(
            "no frame survived the filters; loosen the thresholds and retry"
        )

    accepted.sort(key=lambda item: item[0])
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "pose.txt"), "w", encoding="utf-8") as handle:
        for stamp_ns, xyz, quat in accepted:
            seconds = stamp_ns // 1_000_000_000
            nanos = stamp_ns % 1_000_000_000
            handle.write(
                f"{seconds}.{nanos:09d} "
                f"{xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f} "
                f"{quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}\n"
            )

    spread = np.asarray([a[1] for a in accepted])
    manifest = {
        "source_session": os.path.abspath(args.session) if args.session else None,
        "phone_videos": [os.path.abspath(p) for p in phone_dirs],
        "bag_dirs": [os.path.abspath(b) for b in bag_paths],
        "pose_topic": args.pose_topic,
        "settings": {
            "rotate": args.rotate,
            "max_dt_ms": args.max_dt_ms,
            "max_cov": args.max_cov,
            "min_dist": args.min_dist,
            "min_angle": args.min_angle,
            "min_gap_s": args.min_gap_s,
            "min_sharpness": args.min_sharpness,
            "min_sharpness_quantile": args.min_sharpness_quantile,
            "min_brightness": args.min_brightness,
            "face_min_frac": None if args.no_face_filter else args.face_min_frac,
            "face_detect_scale": (
                None if args.no_face_filter else args.face_detect_scale
            ),
            "face_clahe": None if args.no_face_filter else args.face_clahe,
            "face_min_neighbors": (
                None if args.no_face_filter else args.face_min_neighbors
            ),
        },
        "observed": observed,
        "counts": counts,
        "written": len(accepted),
        "extent_m": {
            "x": [float(spread[:, 0].min()), float(spread[:, 0].max())],
            "y": [float(spread[:, 1].min()), float(spread[:, 1].max())],
        },
    }
    with open(
        os.path.join(args.out, "prepare_manifest.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, indent=2)

    if args.out != final_out:
        shutil.rmtree(final_out, ignore_errors=True)
        os.replace(args.out, final_out)
        args.out = final_out
    print(f"\n[prep] wrote {len(accepted)} frames -> {args.out}")
    print(
        f"[prep] extent x {spread[:, 0].min():.1f}..{spread[:, 0].max():.1f} m, "
        f"y {spread[:, 1].min():.1f}..{spread[:, 1].max():.1f} m"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Build a pose-stamped query recording from phone video + ROS2 bags",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--session", help="capture root; phone videos and bags are found under it"
    )
    parser.add_argument(
        "--phone-video",
        action="append",
        default=[],
        help="explicit phone capture dir (repeatable); overrides discovery",
    )
    parser.add_argument(
        "--bags",
        action="append",
        default=[],
        help="explicit rosbag2 dir OR bare .db3/.mcap (repeatable); overrides discovery",
    )
    parser.add_argument("--out", required=True, help="output recording dir")
    parser.add_argument(
        "--rotate",
        choices=sorted(ROTATIONS),
        default="ccw90",
        help="rotation applied to every frame before anything else",
    )
    parser.add_argument("--pose-topic", default="/pcl_pose")
    parser.add_argument(
        "--max-dt-ms",
        type=float,
        default=50.0,
        help="largest gap allowed between a frame and its bracketing poses",
    )
    parser.add_argument(
        "--max-cov",
        type=float,
        default=0.3,
        help="reject frames whose NDT covariance exceeds this (converged floor is 0.1)",
    )
    parser.add_argument(
        "--min-dist", type=float, default=1.0, help="metres between kept frames"
    )
    parser.add_argument(
        "--min-angle",
        type=float,
        default=15.0,
        help="degrees of heading change that also qualifies as 'moved' "
        "(EITHER gate alone is enough to keep a frame)",
    )
    parser.add_argument(
        "--min-gap-s",
        type=float,
        default=0.15,
        help="never grade two frames closer together in time than this; "
        "assumes ~30 fps footage",
    )
    parser.add_argument(
        "--max-pose-gap-ms",
        type=float,
        default=400.0,
        help="a pose interval wider than this counts as a recording gap "
        "and disables interpolation across it; keep it well above "
        "the pose period (10 Hz -> 100 ms) or frames get snapped "
        "and then dropped as if the bags never covered them",
    )
    parser.add_argument(
        "--allow-unknown-frame-count",
        action="store_true",
        help="proceed when the container reports no frame count, leaving "
        "the meta-index-to-frame mapping unverified",
    )
    parser.add_argument(
        "--max-frame-count-mismatch",
        type=int,
        default=5,
        help="tolerated difference between phone meta records and video frames "
        "before the index mapping is considered untrustworthy",
    )
    parser.add_argument(
        "--min-sharpness",
        type=float,
        default=400.0,
        help="absolute Tenengrad floor. Tenengrad scales with contrast squared, so "
        "a value calibrated on one session does not transfer to another with a "
        "different exposure -- use --stats-only to see this session's spread",
    )
    parser.add_argument(
        "--min-sharpness-quantile",
        type=float,
        default=None,
        help="optionally raise the floor to this quantile of the session's own "
        "sharpness (gate = max(floor, quantile)); costs one extra decode sweep",
    )
    parser.add_argument(
        "--sharpness-sample",
        type=int,
        default=400,
        help="frames sampled across all videos for --min-sharpness-quantile",
    )
    parser.add_argument(
        "--min-brightness",
        type=float,
        default=15.0,
        help="mean grey floor, drops near-black frames",
    )
    parser.add_argument(
        "--face-min-frac",
        type=float,
        default=0.05,
        help="privacy target: reject a frame if a detected face is at least this "
        "tall relative to the image",
    )
    parser.add_argument(
        "--face-detect-scale",
        type=float,
        default=0.5,
        help="scale the frame is detected at, independent of the privacy target. "
        "1.0 runs the cascade well above its 24px native window and cuts false "
        "positives on textured daylight scenes; 0.5 is the low-light default",
    )
    parser.add_argument(
        "--face-clahe",
        choices=("always", "never", "auto"),
        default="always",
        help="contrast-equalise before detection. Needed to find faces in the dark, "
        "counterproductive on well-exposed frames where it amplifies texture",
    )
    parser.add_argument(
        "--face-clahe-below",
        type=float,
        default=80.0,
        help="with --face-clahe auto, equalise only below this mean grey",
    )
    parser.add_argument("--face-min-neighbors", type=int, default=6)
    parser.add_argument("--no-face-filter", action="store_true")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="grade frames and report distributions without writing; pair with "
        "--min-sharpness 0 for an unbiased distribution",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--force", action="store_true", help="overwrite a populated output dir"
    )
    args = parser.parse_args()

    if not args.session and not (args.phone_video and args.bags):
        parser.error("pass --session, or both --phone-video and --bags")
    process(args)


if __name__ == "__main__":
    main()
