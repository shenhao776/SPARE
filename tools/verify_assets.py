#!/usr/bin/env python


import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from core.paper_config import load_paper_config

_PAPER = load_paper_config()
PAPER_SCENES = [
    scene for spec in _PAPER["datasets"].values() for scene in spec["scenes"]
]
STEM_OF = dict(_PAPER["scene_stems"])
DEFAULT_TRAIN_SEED = int(_PAPER["seeds"][0])


_YAML_STEMS = {}


def _stem_from_yaml(scene):

    if not _YAML_STEMS:
        for path in glob.glob(os.path.join(ROOT, "configs", "datasets", "*.yaml")):
            try:
                with open(path, encoding="utf-8") as fh:
                    for line in fh:
                        if line.startswith("name:"):
                            _YAML_STEMS[line.split(":", 1)[1].strip()] = (
                                os.path.splitext(os.path.basename(path))[0]
                            )
                            break
            except OSError:
                pass
    return _YAML_STEMS.get(scene)


def expected_steps(scene):

    from core.config import resolve_dataset

    stem = STEM_OF.get(scene) or _stem_from_yaml(scene)
    if not stem:
        raise KeyError(f"no dataset config for {scene}")
    try:
        seq = resolve_dataset(stem).get("sequence") or {}
        return sum(seq["spatial_steps"]) + 1
    except Exception:
        raise RuntimeError(f"could not resolve sequence config for {scene}")


def _stack_versions():

    out = {}
    for mod in ("torch", "torchvision", "cv2", "numpy", "scipy", "faiss"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = None
    try:
        import torch

        out["gpu"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        )
    except Exception:
        out["gpu"] = None
    return out


class Report:
    def __init__(self):
        self.problems = []
        self.notes = []
        self.facts = {}

    def bad(self, layer, message):
        self.problems.append({"layer": layer, "message": message})

    def note(self, message):

        self.notes.append(message)

    def ok(self, message):
        print(f"  ✅ {message}")


def check_layer1(scenes, report):

    for scene in scenes:
        root = os.path.join(ROOT, "datasets", "database_raw_processed", scene)
        if not os.path.isdir(root):
            report.bad("layer1", f"{scene}: {os.path.relpath(root, ROOT)} is missing")
            continue
        links = [
            os.path.relpath(os.path.join(d, n), ROOT)
            for d, dirs, files in os.walk(root)
            for n in dirs + files
            if os.path.islink(os.path.join(d, n))
        ]
        if links:
            report.bad(
                "layer1",
                f"{scene}: not self-contained, {len(links)} symlinks "
                f"(e.g. {links[0]})",
            )


def check_layer2(scenes, report, deep=False):

    from tools import layout_manifest as lm

    digests = {}
    for scene in scenes:
        manifest = lm.load_manifest(scene)
        if manifest is None:
            report.bad(
                "layer2",
                f"{scene}: no .layout_manifest.json "
                f"(tools/layout_manifest.py --dataset "
                f"{STEM_OF.get(scene, scene)} --write)",
            )
            continue
        digests[scene] = {s: m["digest"] for s, m in manifest["splits"].items()}

        root = os.path.join(ROOT, "datasets", scene)
        for split, recorded in manifest["splits"].items():
            on_disk = dict(lm._split_files(os.path.join(root, split)))
            want = recorded["files"]
            missing = sorted(set(want) - set(on_disk))
            extra = sorted(set(on_disk) - set(want))
            if missing:
                report.bad(
                    "layer2",
                    f"{scene}/{split}: {len(missing)} files in the "
                    f"manifest are gone (e.g. {missing[0]})",
                )
            if extra:
                report.bad(
                    "layer2",
                    f"{scene}/{split}: {len(extra)} files are not in "
                    f"the manifest (e.g. {extra[0]})",
                )
            changed = [
                rel
                for rel in sorted(set(want) & set(on_disk))
                if os.path.getsize(on_disk[rel]) != want[rel][0]
                or (deep and lm.sha256_file(on_disk[rel]) != want[rel][1])
            ]
            if changed:
                report.note(
                    f"{scene}/{split}: {len(changed)} files have the "
                    f"expected name but different bytes (e.g. {changed[0]})"
                )

    report.facts["layer2_digests"] = digests


def check_sequence_assets(scenes, report):
    import torch

    from sl_agent.cache_fingerprint import validate_sequence_asset

    counts = {}
    for scene in scenes:
        db = os.path.join(ROOT, "datasets", scene, "database")
        asset_root = os.path.join(ROOT, "assets", "sequence_assets", scene)
        if not os.path.isdir(asset_root):
            report.bad(
                "assets",
                f"{scene}: no sequence assets at "
                f"{os.path.relpath(asset_root, ROOT)}",
            )
            continue
        n_images = sum(
            1
            for d, _s, files in os.walk(db)
            for f in files
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))
        )
        pts = [
            os.path.join(d, f)
            for d, _s, files in os.walk(asset_root)
            for f in files
            if f.endswith(".pt")
        ]
        leftovers = [
            os.path.join(d, f)
            for d, _s, files in os.walk(asset_root)
            for f in files
            if ".pt.tmp." in f
        ]
        counts[scene] = len(pts)
        if len(pts) != n_images:
            report.bad(
                "assets",
                f"{scene}: {len(pts)} sequence assets for {n_images} "
                "database images",
            )
        if leftovers:
            report.bad(
                "assets",
                f"{scene}: {len(leftovers)} interrupted asset writes "
                f"(e.g. {os.path.relpath(leftovers[0], ROOT)})",
            )
        step = max(1, len(pts) // 10)
        for path in pts[::step][:10]:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            rel_path = os.path.relpath(path, asset_root)[:-3]
            reason = validate_sequence_asset(payload, rel_path, scene)
            if reason:
                report.bad(
                    "assets", f"{scene}: {os.path.relpath(path, ROOT)}: {reason}"
                )
                break
    report.facts["sequence_assets"] = counts


def check_state_caches(report, scenes):

    scoped = set(scenes)
    from sl_agent.cache_fingerprint import cache_stats, structural_error

    summary = {}
    for path in sorted(
        glob.glob(os.path.join(ROOT, "assets", "*", "*_trajectories_*.json"))
    ):
        if path.endswith(".meta.json"):
            continue
        rel = os.path.relpath(path, ROOT)
        base = os.path.basename(path)[: -len(".json")]
        scene = base.split("_trajectories_", 1)[-1]
        for suffix in ("_random_window_no_roi", "_no_roi"):
            if scene.endswith(suffix):
                scene = scene[: -len(suffix)]
                break
        if scene not in scoped:
            continue
        stats = cache_stats(path)
        if stats is None:
            report.bad("state", f"{rel}: unreadable or not a list of rows")
            continue

        meta_path = path + ".meta.json"
        settings = {}
        if os.path.isfile(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as fh:
                    meta = json.load(fh)
                settings = meta.get("settings") or {}
            except (OSError, ValueError):
                report.bad("state", f"{rel}: sidecar is unreadable")
        else:
            report.bad("state", f"{rel}: no fingerprint sidecar")

        want_steps = expected_steps(scene)
        if (
            settings.get("temporal_protocol") != 2
            or settings.get("expected_steps") != want_steps
        ):
            report.bad(
                "state",
                f"{rel}: sidecar does not declare protocol-v2 / " f"{want_steps}-step",
            )

        if settings.get("top_k") is None:
            report.bad("state", f"{rel}: sidecar does not declare top_k")
        broken = structural_error(
            stats,
            {
                "top_k": settings.get("top_k"),
                "expected_steps": settings.get("expected_steps", want_steps),
            },
        )
        if broken:
            report.bad("state", f"{rel}: {broken}")

        summary[rel] = {
            "rows": stats["rows"],
            "queries": len(stats["queries"]),
            "rows_per_query": stats["rows_per_query"],
            "degenerate": stats["degenerate"],
        }
    report.facts["state_caches"] = summary

    orphans = [
        os.path.relpath(p, ROOT)
        for p in glob.glob(os.path.join(ROOT, "assets", "*", "*.shard*"))
    ]
    if orphans:
        report.bad(
            "state",
            f"{len(orphans)} orphaned shard files "
            f"(e.g. {orphans[0]}) -- an interrupted parallel build",
        )


def check_injected_channels(report, scenes):

    from core.config import DINO_K
    from sl_agent.cache_fingerprint import injected_shape_error

    scoped = set(scenes)
    checked = {}
    for path in sorted(
        glob.glob(os.path.join(ROOT, "assets", "*", "*_trajectories_*.json"))
    ):
        if path.endswith(".meta.json"):
            continue
        base = os.path.basename(path)[: -len(".json")]
        scene = base.split("_trajectories_", 1)[-1]
        for suffix in ("_random_window_no_roi", "_no_roi"):
            if scene.endswith(suffix):
                scene = scene[: -len(suffix)]
                break
        if scene not in scoped:
            continue
        rel = os.path.relpath(path, ROOT)
        try:
            window = expected_steps(scene)
        except (KeyError, RuntimeError) as exc:
            report.bad("injected", f"{rel}: {exc}")
            continue
        stamp = {}
        meta_path = path + ".meta.json"
        if os.path.isfile(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as fh:
                    stamp = json.load(fh).get("injectors") or {}
            except (OSError, ValueError):
                pass
        for name, want in (
            ("dino", min(DINO_K, window)),
            ("step_meta", window),
            ("zoom", window),
            ("vprpatch", 1),
        ):
            if name not in stamp:
                continue
            why = injected_shape_error(path, name, want)
            if why:
                report.bad("injected", f"{rel}: {name} {why}")
            else:
                checked[f"{rel}:{name}"] = want
    report.facts["injected_channels"] = checked


def check_weights(report, scenes):

    from train_agent import _training_fingerprint

    checked, unverifiable = 0, 0
    for meta_path in sorted(
        glob.glob(os.path.join(ROOT, "assets", "*", "models", "*.pth.meta.json"))
    ):
        weight = meta_path[: -len(".meta.json")]
        rel = os.path.relpath(weight, ROOT)
        if not any(
            f"_{scene}_" in os.path.basename(weight)
            or os.path.basename(weight).endswith(f"_{scene}.pth")
            for scene in scenes
        ):
            continue
        try:
            with open(meta_path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
        except (OSError, ValueError):
            report.bad("weights", f"{rel}: sidecar unreadable")
            continue
        if meta.get("status") != "complete":
            report.bad(
                "weights",
                f"{rel}: status={meta.get('status')!r} "
                "(an interrupted training run)",
            )
        if not os.path.isfile(weight):
            report.bad("weights", f"{rel}: sidecar without a weight file")
            continue
        from train_agent import _model_output_matches

        if not _model_output_matches(weight, meta):
            report.bad("weights", f"{rel}: weight file is missing or unreadable")
            continue

        settings = meta.get("settings") or {}
        agent_tag, data_tag = settings.get("agent_tag"), settings.get("data_tag")
        seed = settings.get("seed")
        name = os.path.basename(weight)[len("vpr_reranking_gru_") : -len(".pth")]
        seed_suffix = f"_seed{seed}" if seed not in (None, DEFAULT_TRAIN_SEED) else ""
        if not (
            agent_tag
            and name.startswith(agent_tag + "_")
            and name.endswith(data_tag + seed_suffix)
        ):
            unverifiable += 1
            continue
        scene = name[len(agent_tag) + 1 : len(name) - len(data_tag + seed_suffix)]
        model_dir = os.path.dirname(os.path.dirname(weight))
        paths = [
            os.path.join(model_dir, f"{split}_trajectories_{scene}{data_tag}.json")
            for split in ("train", "val")
        ]
        if not all(
            os.path.isfile(p) and os.path.isfile(p + ".meta.json") for p in paths
        ):
            unverifiable += 1
            continue
        try:
            actual = _training_fingerprint(paths[0], paths[1], settings)
        except (FileNotFoundError, ValueError) as exc:
            report.bad("weights", f"{rel}: cannot recompute its fingerprint: {exc}")
            continue
        from train_agent import _legacy_weight_reusable

        if actual != meta.get("training_fingerprint") and not _legacy_weight_reusable(
            meta, paths[0], paths[1], settings
        ):
            report.bad(
                "weights",
                f"{rel}: STALE -- trained on different train/val "
                "state files than the ones on disk now",
            )
        checked += 1
    report.facts["weights_verified"] = checked
    report.facts["weights_unverifiable"] = unverifiable
    print(
        f"     ({checked} weights fingerprint-verified, "
        f"{unverifiable} without their train/val inputs on disk)"
    )


def check_baselines(report):
    files = glob.glob(os.path.join(ROOT, "assets", "paper_baselines", "*.json"))
    report.facts["paper_baselines"] = len(files)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="*", default=PAPER_SCENES)
    ap.add_argument(
        "--deep",
        action="store_true",
        help="re-hash layer ② contents (~50 s for all scenes)",
    )
    ap.add_argument("--json", dest="json_path", help="write a machine-readable report")
    args = ap.parse_args(argv)

    from reproduce import git_provenance

    commit, dirty = git_provenance()
    report = Report()
    report.facts["provenance"] = {
        "commit": commit,
        "dirty": dirty,
        "host": os.uname().nodename if hasattr(os, "uname") else "?",
        "stack": _stack_versions(),
    }

    print(
        f"SPARE asset verification -- commit {commit}"
        f"{' +dirty' if dirty else ''} on {report.facts['provenance']['host']}"
    )
    print("=" * 76)

    for name, fn in (
        ("layer ① portable input", lambda: check_layer1(args.scenes, report)),
        (
            "layer ② evaluation layout",
            lambda: check_layer2(args.scenes, report, args.deep),
        ),
        ("sequence assets", lambda: check_sequence_assets(args.scenes, report)),
        ("state caches", lambda: check_state_caches(report, args.scenes)),
        ("injected channels", lambda: check_injected_channels(report, args.scenes)),
        ("agent weights", lambda: check_weights(report, args.scenes)),
        ("baselines", lambda: check_baselines(report)),
    ):
        before = len(report.problems)
        fn()
        found = len(report.problems) - before
        print(
            f"  {'❌' if found else '✅'} {name}"
            + (f" -- {found} problem(s)" if found else "")
        )

    print("=" * 76)
    for problem in report.problems:
        print(f"[{problem['layer']}] {problem['message']}")
    for note in report.notes:
        print(f"[note] {note}")
    print(f"{len(report.problems)} problem(s), {len(report.notes)} note(s)")

    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as fh:
            json.dump(
                {"problems": report.problems, "notes": report.notes, **report.facts},
                fh,
                indent=2,
                sort_keys=True,
            )
        print(f"report -> {args.json_path}")
    return 1 if report.problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
