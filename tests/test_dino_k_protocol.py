import json
import os
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

os.environ.setdefault("DATASET", "business_campus")

from core.config import DINO_K
from sl_agent.cache_fingerprint import injected_shape_error
from sl_agent.vpr_classifier import normalize_block
from tools.add_dino_channel import _center_slice, _expected_steps
import train_agent


def _rows_json(dir_, seq_len, n=3):
    path = os.path.join(dir_, "cache.json")
    rows = [
        {"query": f"q{i}", "rank": 1, "dino_sim": [0.5] * seq_len} for i in range(n)
    ]
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rows, fh)
    return path


class TestProtocolValue(unittest.TestCase):
    def test_k_is_five_and_odd(self):
        self.assertEqual(DINO_K, 5)
        self.assertEqual(DINO_K % 2, 1)

    def test_injector_expected_steps_is_dino_k(self):
        self.assertEqual(_expected_steps(), DINO_K)


class TestCenterSlice(unittest.TestCase):
    def test_full_window_takes_exact_center(self):
        window = list(range(21))
        self.assertEqual(_center_slice(window), [8, 9, 10, 11, 12])

    def test_short_window_passes_through(self):
        self.assertEqual(_center_slice([1, 2, 3]), [1, 2, 3])
        self.assertEqual(_center_slice([]), [])
        self.assertIsNone(_center_slice(None))

    def test_even_window_still_contains_center_candidate(self):
        window = list(range(8))
        sliced = _center_slice(window)
        self.assertEqual(len(sliced), DINO_K)
        self.assertIn(len(window) // 2, sliced)


class TestNormalizeBlockPath(unittest.TestCase):
    def test_short_channel_left_aligned_zero_padded(self):
        cands = [{"dino_sim": [0.2, 0.4, 0.6, 0.8, 1.0]}]
        out = normalize_block(cands, "dino_sim", 21, required=True)
        self.assertEqual(len(out[0]), 21)
        self.assertEqual([round(v, 6) for v in out[0][:5]], [0.0, 0.25, 0.5, 0.75, 1.0])
        self.assertEqual(out[0][5:], [0.0] * 16)


class TestInjectionGate(unittest.TestCase):

    def _err(self, seq_len):
        with tempfile.TemporaryDirectory() as d:
            return injected_shape_error(_rows_json(d, seq_len), "dino", DINO_K)

    def test_current_shape_skips_rebuild(self):
        self.assertIsNone(self._err(DINO_K))

    def test_legacy_length1_forces_reinject(self):
        self.assertIsNotNone(self._err(1))

    def test_interim_length21_forces_reinject(self):
        self.assertIsNotNone(self._err(21))


class TestTrainingFingerprint(unittest.TestCase):
    def _fp(self, settings):
        with tempfile.TemporaryDirectory() as d:
            t, v = _rows_json(d, DINO_K), _rows_json(d, DINO_K)
            with mock.patch(
                "sl_agent.cache_fingerprint.read_fingerprint",
                return_value={"complete": True, "cache_key": "k", "injectors": {}},
            ):
                return train_agent._training_fingerprint(t, v, settings)

    def test_dino_k_changes_fingerprint(self):
        base = {"agent_tag": "gru", "dino_k": 5}
        self.assertNotEqual(self._fp(base), self._fp({**base, "dino_k": 1}))
        self.assertNotEqual(self._fp(base), self._fp({**base, "dino_k": None}))

    def test_same_settings_same_fingerprint(self):
        base = {"agent_tag": "gru", "dino_k": 5}
        self.assertEqual(self._fp(base), self._fp(dict(base)))


class TestLegacyAdoptionGate(unittest.TestCase):

    def _reusable(self, old_settings, new_settings):
        meta = {"settings": old_settings}
        with mock.patch(
            "sl_agent.cache_fingerprint.read_fingerprint",
            return_value={"migrated_from": "ast-output-sha256"},
        ):
            return train_agent._legacy_weight_reusable(
                meta, "train.json", "val.json", new_settings
            )

    def test_old_sidecar_missing_dino_k_rejected_when_dino_on(self):
        self.assertFalse(self._reusable({"seed": 1}, {"seed": 1, "dino_k": 5}))

    def test_old_sidecar_missing_dino_k_ok_when_dino_off(self):
        self.assertTrue(self._reusable({"seed": 1}, {"seed": 1, "dino_k": None}))

    def test_matching_dino_k_adopts(self):
        self.assertTrue(
            self._reusable({"seed": 1, "dino_k": 5}, {"seed": 1, "dino_k": 5})
        )

    def test_stale_dino_k_rejected(self):
        self.assertFalse(
            self._reusable({"seed": 1, "dino_k": 1}, {"seed": 1, "dino_k": 5})
        )

    def test_other_setting_drift_still_rejected(self):
        self.assertFalse(
            self._reusable({"seed": 1, "dino_k": 5}, {"seed": 2, "dino_k": 5})
        )


if __name__ == "__main__":
    unittest.main()
