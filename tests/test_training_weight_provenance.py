import json
import os
import tempfile
import unittest

from tests.fingerprint_helpers import perturb as _perturb

from train_agent import (
    _legacy_weight_reusable,
    _model_output_matches,
    _training_fingerprint,
)


class TrainingWeightProvenanceTests(unittest.TestCase):
    def test_atomic_weight_presence_is_the_fast_integrity_gate(self):
        with tempfile.TemporaryDirectory() as temp:
            path = os.path.join(temp, "agent.pth")
            with open(path, "wb") as handle:
                handle.write(b"weight-v1")
            self.assertTrue(_model_output_matches(path, {"status": "complete"}))

            with open(path, "wb") as handle:
                handle.write(b"weight-v2")
            self.assertTrue(_model_output_matches(path, {"status": "complete"}))

    def test_missing_weight_is_not_reused(self):
        with tempfile.TemporaryDirectory() as temp:
            path = os.path.join(temp, "agent.pth")
            self.assertFalse(_model_output_matches(path, {"status": "complete"}))

    def test_training_key_tracks_state_keys_and_injected_channels(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = [os.path.join(temp, f"{split}.json") for split in ("train", "val")]
            for index, path in enumerate(paths):
                with open(path, "w", encoding="utf-8") as handle:
                    json.dump([], handle)
                with open(path + ".meta.json", "w", encoding="utf-8") as handle:
                    json.dump(
                        {
                            "complete": True,
                            "cache_key": f"state-{index}",
                            "injectors": {"zoom": {"content_version": 1}},
                        },
                        handle,
                    )
            first = _training_fingerprint(*paths, {"content_version": 1})
            with open(paths[1] + ".meta.json", "r+", encoding="utf-8") as handle:
                stamp = json.load(handle)
                stamp["injectors"]["zoom"]["content_version"] = 2
                handle.seek(0)
                json.dump(stamp, handle)
                handle.truncate()
            second = _training_fingerprint(*paths, {"content_version": 1})
            self.assertNotEqual(first, second)

    def test_训练指纹的每个设置改了都必须换指纹(self):

        base = {
            "protocol": 3,
            "seed": 2026,
            "input_dim": 17,
            "hidden_dim": 64,
            "epochs": 50,
            "batch_size": 32,
            "learning_rate": 0.001,
            "spatial_steps": [10, 10],
            "success_threshold_train": 8.0,
            "success_threshold_test": 5.0,
            "agent_tag": "arch=gru_deriv=True",
            "data_tag": "_no_roi",
            "dino_k": 5,
        }
        with tempfile.TemporaryDirectory() as temp:
            paths = [os.path.join(temp, f"{split}.json") for split in ("train", "val")]
            for path in paths:
                with open(path, "w", encoding="utf-8") as handle:
                    json.dump([], handle)
                with open(path + ".meta.json", "w", encoding="utf-8") as handle:
                    json.dump({"complete": True, "cache_key": "k"}, handle)
            reference = _training_fingerprint(*paths, base)
            for key, value in sorted(base.items()):
                with self.subTest(key=key):
                    changed = dict(base)
                    changed[key] = _perturb(value)
                    self.assertNotEqual(changed[key], value)
                    self.assertNotEqual(
                        reference,
                        _training_fingerprint(*paths, changed),
                        f"改了 {key} 却得到同一个训练指纹 —— 旧权重会被无声复用",
                    )

    def test_训练指纹字段清单与生产代码一致(self):

        import re

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidates = [
            os.path.join(root, "pipeline", "train_agent.py"),
            os.path.join(root, "train_agent.py"),
        ]
        source = open(
            next(path for path in candidates if os.path.exists(path)),
            encoding="utf-8",
        ).read()
        block = re.search(r"fingerprint_settings = \{(.*?)\n    \}", source, re.S)
        self.assertIsNotNone(block, "找不到 fingerprint_settings 定义块")
        keys = set(re.findall(r'^\s*"([a-z_]+)":', block.group(1), re.M))
        covered = {
            "protocol",
            "seed",
            "input_dim",
            "hidden_dim",
            "epochs",
            "batch_size",
            "learning_rate",
            "spatial_steps",
            "success_threshold_train",
            "success_threshold_test",
            "agent_tag",
            "data_tag",
            "dino_k",
        }
        self.assertEqual(
            keys,
            covered,
            "train_agent.fingerprint_settings 的字段变了,请同步更新上一条遍历测试的样例值",
        )

    def test_weight_sidecar_migrates_after_preprocessing_key_removal(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = [os.path.join(temp, f"{split}.json") for split in ("train", "val")]
            for path in paths:
                with open(path, "w", encoding="utf-8") as handle:
                    json.dump([], handle)
                with open(path + ".meta.json", "w", encoding="utf-8") as handle:
                    json.dump(
                        {
                            "complete": True,
                            "cache_key": "new-direct-settings-key",
                            "migrated_from": "preprocessing-digest-key",
                        },
                        handle,
                    )
            settings = {"content_version": 1, "seed": 2026}
            self.assertTrue(
                _legacy_weight_reusable(
                    {"settings": settings}, paths[0], paths[1], settings
                )
            )


if __name__ == "__main__":
    unittest.main()
