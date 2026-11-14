import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("DATASET", "robotcar")

import reproduce


def _scenes_with_manifest():
    out = []
    base = os.path.join(ROOT, "datasets")
    if not os.path.isdir(base):
        return out
    from core.paper_config import load_paper_config

    stems = load_paper_config()["scene_stems"]
    for scene, stem in stems.items():
        if os.path.isfile(os.path.join(base, scene, ".layout_manifest.json")):
            out.append((scene, stem))
    return out


class TestLayoutParamGate(unittest.TestCase):
    def test_unchanged_scenes_do_not_false_positive(self):

        scenes = _scenes_with_manifest()
        if not scenes:
            self.skipTest("本机没有带 manifest 的场景")
        for scene, stem in scenes:
            with self.subTest(scene=scene):
                diff = reproduce._layout_params_changed(scene, stem)
                if scene == "Boreas":
                    continue
                self.assertEqual(diff, "", f"{scene} 被误判为参数变更:{diff}")

    def test_changed_interval_is_detected(self):

        scenes = _scenes_with_manifest()
        if not scenes:
            self.skipTest("本机没有带 manifest 的场景")
        scene, stem = scenes[0]
        real = reproduce._layout_params_changed

        from tools import layout_manifest as lm

        original = lm.load_manifest

        def fake(name):
            got = original(name)
            if got and got.get("params", {}).get("build"):
                got["params"]["build"]["intervals"] = {
                    "train": 99,
                    "val": 99,
                    "test": 99,
                }
            return got

        lm.load_manifest = fake
        try:
            diff = real(scene, stem)
        finally:
            lm.load_manifest = original
        self.assertIn("intervals", diff)

    def test_missing_manifest_is_not_treated_as_change(self):

        self.assertEqual(
            reproduce._layout_params_changed("__NoSuchScene__", "robotcar"), ""
        )


class TestNoSilentSceneFallback(unittest.TestCase):

    def test_worker_without_dataset_fails_loudly(self):
        import subprocess

        env = {
            k: v for k, v in os.environ.items() if k not in ("DATASET", "TARGET_SCENE")
        }
        out = subprocess.run(
            [sys.executable, "-c", "import core.config"],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertNotEqual(out.returncode, 0, "DATASET 未设却成功导入了 core.config")
        self.assertIn("必须指定场景", out.stdout + out.stderr)

    def test_orchestrators_still_import(self):

        import subprocess

        env = {
            k: v for k, v in os.environ.items() if k not in ("DATASET", "TARGET_SCENE")
        }
        entries = []
        for flat, nested in (
            ("reproduce.py", "pipeline/reproduce.py"),
            ("run_paper_tables.py", "replay/run_paper_tables.py"),
            ("tools/run_crop_sweep.py", "tools/run_crop_sweep.py"),
        ):
            entries.append(
                nested if os.path.exists(os.path.join(ROOT, nested)) else flat
            )
        for entry in entries:
            with self.subTest(entry=entry):
                out = subprocess.run(
                    [sys.executable, entry, "--help"],
                    cwd=ROOT,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=180,
                )
                self.assertEqual(
                    out.returncode, 0, f"{entry} --help 挂了:{out.stderr[-400:]}"
                )

    def test_default_yaml_has_no_dataset_fallback(self):
        import yaml

        with open(
            os.path.join(ROOT, "configs", "default.yaml"), encoding="utf-8"
        ) as fh:
            cfg = yaml.safe_load(fh)
        self.assertNotIn(
            "dataset",
            cfg.get("defaults", {}),
            "configs/default.yaml 又出现了 defaults.dataset 兜底 —— 不要加回来",
        )


if __name__ == "__main__":
    unittest.main()
