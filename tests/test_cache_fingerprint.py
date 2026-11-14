import json
import os
import tempfile
import unittest
from unittest import mock

from sl_agent import cache_fingerprint as fp


class CacheFingerprintTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        fp._FILE_DIGESTS.clear()

    def path(self, name):
        return os.path.join(self.temp.name, name)

    def write_query(self, name="query.jpg", payload=b"abcd"):
        path = self.path(name)
        with open(path, "wb") as handle:
            handle.write(payload)
        return path

    def write_valid_cache(self, query, settings=None):
        output = self.path("state.json")
        row = {
            "query": "q",
            "inliers": [1, 2],
            "scores": [0.1, 0.2],
            "step_errors": [1.0, 2.0],
            "area_step": [0.2, 0.3],
            "area_query": [0.4, 0.5],
            "scale_ratio": [1.0, 1.1],
        }
        with open(output, "w", encoding="utf-8") as handle:
            json.dump([row], handle)
        settings = settings or {"top_k": 1, "expected_steps": 2}
        stamp = fp.build_fingerprint([query], settings)
        fp.write_fingerprint(output, stamp)
        return output, stamp

    def test_setting_change_invalidates_cache(self):
        query = self.write_query()
        output, _ = self.write_valid_cache(query, {"top_k": 1, "expected_steps": 2})
        changed = fp.build_fingerprint(
            [query], {"top_k": 1, "expected_steps": 2, "step_max_dist": 3.0}
        )
        self.assertIn("content key changed", fp.resume_reason(output, changed))

    def test_removed_preprocessing_digests_migrate_without_json_rewrite(self):
        query = self.write_query()
        old_settings = {
            "top_k": 1,
            "expected_steps": 2,
            "walk_sha256": "old-walk",
            "layout_query_digest": "old-layout",
            "sequence_assets": {"semantic_digest": "old-assets"},
        }
        output, _ = self.write_valid_cache(query, old_settings)
        before = os.stat(output)
        current = fp.build_fingerprint([query], {"top_k": 1, "expected_steps": 2})
        self.assertIsNone(fp.resume_reason(output, current))
        after = os.stat(output)
        self.assertEqual(
            (before.st_size, before.st_mtime_ns), (after.st_size, after.st_mtime_ns)
        )
        self.assertEqual(
            fp.read_fingerprint(output)["migrated_from"],
            "preprocessing-digest-key",
        )

    def test_unrelated_query_mtime_and_bytes_are_not_hashed(self):
        query = self.write_query(payload=b"AAAA")
        output, _ = self.write_valid_cache(query)
        old_stat = os.stat(query)
        with open(query, "wb") as handle:
            handle.write(b"BBBB")
        os.utime(query, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
        changed = fp.build_fingerprint([query], {"top_k": 1, "expected_steps": 2})
        self.assertIsNone(fp.resume_reason(output, changed))

    def test_minimal_legacy_sidecar_is_migrated_without_json_rewrite(self):
        query = self.write_query()
        output, stamp = self.write_valid_cache(query)
        before = os.stat(output)
        legacy = {
            "complete": True,
            "n_queries": stamp["n_queries"],
            "settings": {"top_k": 1, "expected_steps": 2},
        }
        with open(fp.meta_path(output), "w", encoding="utf-8") as handle:
            json.dump(legacy, handle)
        self.assertIsNone(fp.resume_reason(output, stamp))
        after = os.stat(output)
        self.assertEqual(
            (before.st_size, before.st_mtime_ns), (after.st_size, after.st_mtime_ns)
        )
        self.assertEqual(
            fp.read_fingerprint(output)["migrated_from"], "legacy-v1-minimal"
        )

    def test_resume_does_not_hash_or_parse_large_output_json(self):
        query = self.write_query()
        output, stamp = self.write_valid_cache(query)
        with open(output, encoding="utf-8") as handle:
            rows = json.load(handle)
        rows[0]["scores"] = [0.9, 0.8]
        with open(output, "w", encoding="utf-8") as handle:
            json.dump(rows, handle)
        self.assertIsNone(fp.resume_reason(output, stamp))

    def test_postprocessor_can_atomically_refresh_output_identity(self):
        query = self.write_query()
        output, stamp = self.write_valid_cache(query)
        with open(output, encoding="utf-8") as handle:
            rows = json.load(handle)
        rows[0]["zoom"] = [2.0, 2.0]
        with open(output, "w", encoding="utf-8") as handle:
            json.dump(rows, handle)
        fp.refresh_output_fingerprint(output, zoom={"protocol": 1})
        self.assertIsNone(fp.resume_reason(output, stamp))
        sidecar = fp.read_fingerprint(output)
        self.assertEqual(sidecar["injectors"]["zoom"], {"protocol": 1})

    def test_old_ast_injector_stamp_migrates_without_rewriting_json(self):
        query = self.write_query()
        output, _ = self.write_valid_cache(query)
        stamp = fp.read_fingerprint(output)
        stamp["injectors"] = {
            "zoom": {"content_version": 1, "protocol": 1, "producer_ast_sha256": "old"}
        }
        with open(fp.meta_path(output), "w", encoding="utf-8") as handle:
            json.dump(stamp, handle)
        before = os.stat(output)
        self.assertTrue(
            fp.injector_current(output, "zoom", {"content_version": 1, "protocol": 1})
        )
        after = os.stat(output)
        self.assertEqual(
            (before.st_size, before.st_mtime_ns), (after.st_size, after.st_mtime_ns)
        )

    def test_fixed_and_auto_artifact_names_are_disjoint(self):
        from core.config import data_tag

        self.assertEqual(data_tag(step_max_tag=""), "_no_roi")
        self.assertEqual(data_tag(step_max_tag="auto"), "_stepmax-auto_no_roi")

    def test_sequence_asset_requires_structure_and_exact_frame(self):
        payload = {
            "format": "lean-v2",
            "provenance_digest": "sig",
            "source_rel_path": "database/1.jpg",
            "rel_seq_paths": ["database/1.jpg"],
            "lightglue_feats": [{"keypoints": object()}],
        }
        self.assertIsNone(
            fp.validate_sequence_asset(payload, "database/1.jpg", "RobotCar")
        )
        self.assertIn(
            "source_rel_path",
            fp.validate_sequence_asset(payload, "database/2.jpg", "RobotCar"),
        )
        legacy = dict(payload)
        legacy.pop("provenance_digest")
        self.assertIsNone(
            fp.validate_sequence_asset(legacy, "database/1.jpg", "RobotCar")
        )


if __name__ == "__main__":
    unittest.main()
