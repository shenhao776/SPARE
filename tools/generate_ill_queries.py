import os
import argparse
import math
import cv2
import numpy as np
import sys
import random
import shutil
import zlib
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config import CFG

_crop = getattr(getattr(CFG.dataset, "build", None), "crop", None)
_crop_default = CFG.query_crop


def _split_crop(crop):

    if os.getenv("QUERY_SPLIT", "") != "test" or crop is None:
        return crop
    over = getattr(crop, "crop_test", None) or getattr(
        getattr(CFG.dataset, "build", None), "crop_test", None
    )
    if over is None:
        return crop
    import types

    merged = dict(vars(crop))
    merged.update({k: v for k, v in vars(over).items()})
    return types.SimpleNamespace(**merged)


_crop = _split_crop(_crop)
MIN_CROP_RATIO = float(
    os.getenv("CROP_MIN_RATIO", getattr(_crop, "min_ratio", _crop_default.min_ratio))
)
MAX_CROP_RATIO = float(
    os.getenv("CROP_MAX_RATIO", getattr(_crop, "max_ratio", _crop_default.max_ratio))
)
CROP_TILT_DEG = float(
    os.getenv("CROP_TILT_DEG", getattr(_crop, "tilt_deg", _crop_default.tilt_deg))
)
CROP_CENTERED = (
    os.getenv(
        "CROP_CENTERED",
        "1" if getattr(_crop, "centered", _crop_default.centered) else "0",
    )
    == "1"
)
CROP_SEED = int(os.getenv("CROP_SEED", getattr(_crop, "seed", _crop_default.seed)))
CROP_WARP_DEG = float(
    os.getenv("CROP_WARP_DEG", getattr(_crop, "warp_deg", _crop_default.warp_deg))
)
CROP_WARP_FOCAL_RATIO = float(
    os.getenv(
        "CROP_WARP_FOCAL_RATIO",
        getattr(_crop, "warp_focal_ratio", _crop_default.warp_focal_ratio),
    )
)

TOP_EXCL = float(
    os.getenv("CROP_TOP_EXCL", getattr(_crop, "top_excl", 0.0) if _crop else 0.0)
)
BOTTOM_EXCL = float(
    os.getenv("CROP_BOTTOM_EXCL", getattr(_crop, "bottom_excl", 0.3) if _crop else 0.3)
)

_pseudo_aug_raw = os.getenv(
    "PSEUDO_AUG", getattr(_crop, "pseudo_aug", _crop_default.pseudo_aug)
)
if _pseudo_aug_raw is False or _pseudo_aug_raw is None:
    PSEUDO_AUG = "off"
elif _pseudo_aug_raw is True:
    raise SystemExit("pseudo_aug=true is ambiguous; choose off/light/medium/strong")
else:
    PSEUDO_AUG = str(_pseudo_aug_raw).strip().lower()
_AUG_LEVELS = {
    "light": (
        (0.75, 0.95),
        (0.0, 0.6),
        (0.90, 1.10),
        0.04,
        (0.90, 1.10),
        (0, 3),
        (75, 92),
        1.5,
    ),
    "medium": (
        (0.55, 0.85),
        (0.4, 1.2),
        (0.80, 1.25),
        0.08,
        (0.82, 1.20),
        (2, 6),
        (55, 85),
        3.0,
    ),
    "strong": (
        (0.40, 0.75),
        (0.6, 1.8),
        (0.70, 1.40),
        0.12,
        (0.75, 1.30),
        (3, 9),
        (40, 80),
        5.0,
    ),
}
if PSEUDO_AUG not in {"off", *_AUG_LEVELS}:
    raise SystemExit(
        f"invalid pseudo_aug={PSEUDO_AUG!r}; choose off/light/medium/strong"
    )


def simulate_human_capture(img, target_w, target_h, top_excl, bottom_excl, rng=random):

    h, w = img.shape[:2]

    min_y = int(h * top_excl)
    max_y_bound = int(h * (1.0 - bottom_excl))
    valid_h = max_y_bound - min_y

    max_x = w - target_w
    max_y = valid_h - target_h

    start_x = rng.randint(0, max_x) if max_x > 0 else 0
    start_y = min_y + (rng.randint(0, max_y) if max_y > 0 else 0)

    end_y = min(start_y + target_h, h)
    end_x = min(start_x + target_w, w)

    cropped = img[start_y:end_y, start_x:end_x]

    angle = rng.uniform(-CROP_TILT_DEG, CROP_TILT_DEG)
    crop_h, crop_w = cropped.shape[:2]
    M = cv2.getRotationMatrix2D((crop_w / 2, crop_h / 2), angle, 1.0)
    if CROP_WARP_DEG <= 0:
        rotated = cv2.warpAffine(
            cropped, M, (crop_w, crop_h), borderMode=cv2.BORDER_REPLICATE
        )
    else:
        yaw = math.radians(rng.uniform(-CROP_WARP_DEG, CROP_WARP_DEG))
        pitch = math.radians(rng.uniform(-CROP_WARP_DEG, CROP_WARP_DEG))
        roll = math.radians(angle)
        cy_, sy_ = math.cos(yaw), math.sin(yaw)
        cp_, sp_ = math.cos(pitch), math.sin(pitch)
        cr_, sr_ = math.cos(roll), math.sin(roll)
        R = (
            np.float64([[cr_, -sr_, 0], [sr_, cr_, 0], [0, 0, 1]])
            @ np.float64([[cy_, 0, sy_], [0, 1, 0], [-sy_, 0, cy_]])
            @ np.float64([[1, 0, 0], [0, cp_, -sp_], [0, sp_, cp_]])
        )
        f = max(1.0, CROP_WARP_FOCAL_RATIO * crop_w)
        K = np.float64([[f, 0, crop_w / 2.0], [0, f, crop_h / 2.0], [0, 0, 1]])
        H = K @ R @ np.linalg.inv(K)
        T = np.float64([[1, 0, start_x], [0, 1, start_y], [0, 0, 1]])
        src_from_out = T @ H

        corners = np.float64(
            [[0, 0, 1], [crop_w, 0, 1], [crop_w, crop_h, 1], [0, crop_h, 1]]
        ).T

        def _fit(scale, tx, ty):
            S = np.float64(
                [
                    [scale, 0, (1 - scale) * crop_w / 2.0 + tx],
                    [0, scale, (1 - scale) * crop_h / 2.0 + ty],
                    [0, 0, 1],
                ]
            )
            pts = src_from_out @ S @ corners
            return S, pts[:2] / pts[2]

        scale, tx, ty = 1.0, 0.0, 0.0
        for _ in range(24):
            _, pts = _fit(scale, tx, ty)
            tx += -min(0.0, pts[0].min()) - max(0.0, pts[0].max() - (w - 1))
            ty += -min(0.0, pts[1].min()) - max(0.0, pts[1].max() - (h - 1))
            _, pts = _fit(scale, tx, ty)
            if (
                pts[0].min() >= 0
                and pts[1].min() >= 0
                and pts[0].max() <= w - 1
                and pts[1].max() <= h - 1
            ):
                break
            scale *= 0.97
        else:
            scale, tx, ty = 1.0, 0.0, 0.0
        S, _ = _fit(scale, tx, ty)
        rotated = cv2.warpPerspective(
            img,
            src_from_out @ S,
            (crop_w, crop_h),
            flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REPLICATE,
        )

    telephoto_img = cv2.resize(rotated, (w, h), interpolation=cv2.INTER_CUBIC)
    return telephoto_img, start_x, start_y


def apply_capture_augmentation(img, rng, level):

    cfg = _AUG_LEVELS.get(level)
    if cfg is None:
        return img
    (
        (ds_lo, ds_hi),
        (bl_lo, bl_hi),
        (g_lo, g_hi),
        wb,
        (c_lo, c_hi),
        (n_lo, n_hi),
        (q_lo, q_hi),
        persp,
    ) = cfg
    h, w = img.shape[:2]

    if persp > 0:
        d = math.tan(math.radians(rng.uniform(-persp, persp)))
        e = math.tan(math.radians(rng.uniform(-persp, persp)))
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = np.float32([[d * w, e * h], [w - d * w, 0], [w, h - e * h], [0, h]])
        img = cv2.warpPerspective(
            img,
            cv2.getPerspectiveTransform(src, dst),
            (w, h),
            borderMode=cv2.BORDER_REPLICATE,
        )

    f = rng.uniform(ds_lo, ds_hi)
    small = cv2.resize(
        img, (max(8, int(w * f)), max(8, int(h * f))), interpolation=cv2.INTER_AREA
    )
    img = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    sigma = rng.uniform(bl_lo, bl_hi)
    if sigma > 0.05:
        img = cv2.GaussianBlur(img, (0, 0), sigma)

    x = img.astype(np.float32) / 255.0
    x = np.power(np.clip(x, 0, 1), rng.uniform(g_lo, g_hi))
    for ch in range(3):
        x[:, :, ch] *= 1.0 + rng.uniform(-wb, wb)
    contrast = rng.uniform(c_lo, c_hi)
    x = (x - x.mean()) * contrast + x.mean()
    img = np.clip(x * 255.0, 0, 255).astype(np.uint8)

    n = rng.uniform(n_lo, n_hi)
    if n > 0.1:
        noise = (
            np.random.default_rng(rng.getrandbits(64))
            .normal(0.0, n, img.shape)
            .astype(np.float32)
        )
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    ok, buf = cv2.imencode(
        ".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.uniform(q_lo, q_hi))]
    )
    if ok:
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return img


def standard_robot_crop(img, target_w, target_h, top_excl, bottom_excl):

    h, w = img.shape[:2]

    min_y = int(h * top_excl)
    max_y_bound = int(h * (1.0 - bottom_excl))
    valid_h = max_y_bound - min_y

    start_x = (w - target_w) // 2
    start_y = min_y + (valid_h - target_h) // 2

    end_y = min(start_y + target_h, h)
    end_x = min(start_x + target_w, w)

    cropped = img[start_y:end_y, start_x:end_x]

    telephoto_img = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_CUBIC)
    return telephoto_img, start_x, start_y


def process_single_file(
    file_path,
    input_dir,
    output_dir,
    debug_dir,
    valid_img_exts,
    top_excl,
    bottom_excl,
    full_frame_recordings=(),
):
    rel_path = os.path.relpath(file_path, input_dir)
    save_path = os.path.join(output_dir, rel_path)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if not file_path.lower().endswith(valid_img_exts):
        shutil.copy2(file_path, save_path)
        return

    top_rec = rel_path.replace("\\", "/").split("/")[0]
    if top_rec in full_frame_recordings:
        shutil.copy2(file_path, save_path)
        if debug_dir:
            dbg = os.path.join(debug_dir, rel_path)
            os.makedirs(os.path.dirname(dbg), exist_ok=True)
            shutil.copy2(file_path, dbg)
        return

    img = cv2.imread(file_path)
    if img is None:
        shutil.copy2(file_path, save_path)
        return

    h, w = img.shape[:2]

    rng = random.Random(zlib.crc32(rel_path.encode("utf-8")) ^ CROP_SEED)

    dynamic_ratio = rng.uniform(MIN_CROP_RATIO, MAX_CROP_RATIO)

    if dynamic_ratio >= 1.0:
        shutil.copy2(file_path, save_path)
        if debug_dir:
            dbg = os.path.join(debug_dir, rel_path)
            os.makedirs(os.path.dirname(dbg), exist_ok=True)
            shutil.copy2(file_path, dbg)
        return

    crop_w = int(w * dynamic_ratio)
    crop_h = int(h * dynamic_ratio)

    if CROP_CENTERED:
        res_img, start_x, start_y = standard_robot_crop(
            img, crop_w, crop_h, top_excl, bottom_excl
        )
    else:
        res_img, start_x, start_y = simulate_human_capture(
            img, crop_w, crop_h, top_excl, bottom_excl, rng
        )

    res_img = apply_capture_augmentation(res_img, rng, PSEUDO_AUG)

    cv2.imwrite(save_path, res_img)

    if debug_dir:
        debug_img = img.copy()

        if top_excl > 0:
            top_y = int(h * top_excl)
            cv2.line(debug_img, (0, top_y), (w, top_y), (255, 0, 0), 3)
            cv2.putText(
                debug_img,
                f"Top Exclusion ({top_excl*100:.0f}%)",
                (20, top_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (255, 0, 0),
                3,
            )

        if bottom_excl > 0:
            bottom_y = int(h * (1.0 - bottom_excl))
            cv2.line(debug_img, (0, bottom_y), (w, bottom_y), (255, 0, 0), 3)

        cv2.rectangle(
            debug_img,
            (start_x, start_y),
            (start_x + crop_w, start_y + crop_h),
            (0, 0, 255),
            5,
        )

        debug_save_path = os.path.join(debug_dir, rel_path)
        os.makedirs(os.path.dirname(debug_save_path), exist_ok=True)
        cv2.imwrite(debug_save_path, debug_img)


def generate_telephoto_queries(
    input_dir,
    output_dir,
    debug_dir=None,
    top_excl=0.0,
    bottom_excl=0.3,
    max_workers=12,
    full_frame_recordings=(),
):
    if max_workers is None:
        max_workers = os.cpu_count() or 4

    if top_excl + bottom_excl >= 1.0:
        print("❌ Error: top + bottom exclusion ratios must be < 1.0")
        return

    valid_ratio = 1.0 - top_excl - bottom_excl
    if MAX_CROP_RATIO > valid_ratio:
        print(
            f"⚠️ Warning: MAX_CROP_RATIO ({MAX_CROP_RATIO}) exceeds valid height ratio ({valid_ratio:.2f}); the crop box will be clamped."
        )

    print("=" * 65)
    print("📸 SPARE: starting telephoto query generator (mixed difficulty pool)")
    print(
        f"⚙️  Config -> crop ratio: {MIN_CROP_RATIO}% ~ {MAX_CROP_RATIO}% | top excl: {top_excl*100}% | bottom excl: {bottom_excl*100}% | workers: {max_workers}"
    )
    print("=" * 65)

    if not os.path.exists(input_dir):
        print(f"❌ Input directory not found: {input_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)

    all_files = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            all_files.append(os.path.join(root, f))

    if not all_files:
        print("❌ No files found under input directory.")
        return

    print(f"📦 Found {len(all_files)} files. Starting processing...")

    valid_img_exts = (".png", ".jpg", ".jpeg", ".tiff")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(
                process_single_file,
                file_path,
                input_dir,
                output_dir,
                debug_dir,
                valid_img_exts,
                top_excl,
                bottom_excl,
                full_frame_recordings,
            ): file_path
            for file_path in all_files
        }

        for future in tqdm(
            as_completed(future_to_file), total=len(all_files), desc="Processing files"
        ):
            file_path = future_to_file[future]
            try:
                future.result()
            except Exception as exc:
                print(f"\n❌ Exception while processing {file_path}: {exc}")

    print(f"\n✅ Done. Query images saved to: {output_dir}")
    if debug_dir:
        print(f"✅ Debug images (crop box + exclusion zones) saved to: {debug_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate query images with random difficulty (mixed pool)"
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Input query directory"
    )
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument(
        "--debug", type=str, default=None, help="Debug output directory (red crop box)"
    )
    parser.add_argument(
        "--crop_ratio",
        type=float,
        default=0.3,
        help="(deprecated) now uses random dynamic ratio",
    )
    parser.add_argument(
        "--top_excl",
        type=float,
        default=TOP_EXCL,
        help="Top exclusion ratio (e.g., 0.2 means avoid top 20%)",
    )
    parser.add_argument(
        "--bottom_excl",
        type=float,
        default=BOTTOM_EXCL,
        help="Bottom exclusion ratio (e.g., 0.3 means avoid bottom 30%)",
    )
    parser.add_argument(
        "--workers", type=int, default=None, help="Number of worker threads"
    )
    parser.add_argument(
        "--full-frame-recordings",
        type=str,
        default="",
        help="comma-separated recording names that skip the synthetic crop and "
        "use the WHOLE original image (e.g. real optical-zoom captures)",
    )

    random.seed(CROP_SEED)

    args = parser.parse_args()
    full_frame = {r for r in args.full_frame_recordings.split(",") if r}

    if args.input == args.output:
        print("❌ Error: input and output directories must be different")
    else:
        generate_telephoto_queries(
            args.input,
            args.output,
            args.debug,
            args.top_excl,
            args.bottom_excl,
            args.workers,
            full_frame_recordings=full_frame,
        )
