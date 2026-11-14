import copy
import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

from core import checkpoint_provenance
from tools.rerank_eval_common import matching_result, result_metadata, save_result


class RerankResultMetadataTests(unittest.TestCase):
    def _inputs(self, root):
        cache = os.path.join(root, "pool.json")
        weight = os.path.join(root, "model.pth")
        with open(cache, "wb") as handle:
            handle.write(b"AAAA")
        with open(weight, "wb") as handle:
            handle.write(b"weights")
        return cache, weight

    def _saved_result(self, root, metadata, thresholds=(5.0, 10.0, 15.0)):
        path = os.path.join(root, "result.json")
        save_result(
            path,
            metadata,
            thresholds,
            raw_errors=[1.0, 12.0],
            reranked_errors=[2.0, 8.0],
            selected_ranks=[1, 2],
            elapsed_s=123.0,
        )
        return path

    def test_same_size_same_mtime_input_replacement_changes_identity(self):
        with tempfile.TemporaryDirectory() as root:
            cache, weight = self._inputs(root)
            digest_cache = os.path.join(root, "digests.json")

            with mock.patch.object(checkpoint_provenance, "_CACHE_FILE", digest_cache):
                original = result_metadata("method", "scene", cache, [weight])
                old = os.stat(cache)
                with open(cache, "wb") as handle:
                    handle.write(b"BBBB")
                os.utime(cache, ns=(old.st_atime_ns, old.st_mtime_ns))
                changed = result_metadata("method", "scene", cache, [weight])

            self.assertNotEqual(original["cache"]["sha256"], changed["cache"]["sha256"])
            self.assertNotIn("mtime_ns", changed["cache"])
            self.assertNotIn("path", changed["cache"])
            self.assertEqual(original["checkpoints"], changed["checkpoints"])

    def test_legacy_version_and_mtime_metadata_migrate_without_evaluation(self):
        with tempfile.TemporaryDirectory() as root:
            cache, weight = self._inputs(root)
            current = result_metadata(
                "Method",
                "Scene",
                cache,
                [weight],
                {"resize": [322, 322], "n_queries_limit": 0},
            )
            legacy = copy.deepcopy(current)
            legacy["evaluator_version"] = -999
            legacy["cache"] = {
                "path": cache,
                "size": os.path.getsize(cache),
                "mtime_ns": 1,
            }
            legacy["checkpoints"] = [
                {
                    "path": weight,
                    "size": os.path.getsize(weight),
                    "mtime_ns": 2,
                }
            ]
            legacy["extra"].update(
                {
                    "backbone": "description only",
                    "official_checkpoint": True,
                }
            )
            path = self._saved_result(root, legacy)

            migrated = matching_result(path, current, thresholds=(5.0, 15.0))

            self.assertIsNotNone(migrated)
            self.assertEqual(migrated["metadata"], current)
            self.assertEqual(migrated["thresholds_m"], [5.0, 15.0])
            self.assertEqual(migrated["raw_recall"], {"5": 50.0, "15": 100.0})
            with open(path, encoding="utf-8") as handle:
                persisted = json.load(handle)
            self.assertEqual(persisted["metadata"], current)
            self.assertNotIn("evaluator_version", persisted["metadata"])

    def test_only_direct_scoring_settings_invalidate_result(self):
        with tempfile.TemporaryDirectory() as root:
            cache, weight = self._inputs(root)
            original = result_metadata(
                "Method",
                "Scene",
                cache,
                [weight],
                {"radius": 0.1, "n_queries_limit": 0},
            )
            path = self._saved_result(root, original)

            descriptive = copy.deepcopy(original)
            descriptive["extra"].update(
                {
                    "backbone": "renamed description",
                    "official_checkpoint": False,
                }
            )
            self.assertIsNotNone(matching_result(path, descriptive))

            changed = result_metadata(
                "Method",
                "Scene",
                cache,
                [weight],
                {"radius": 0.2, "n_queries_limit": 0},
            )
            self.assertIsNone(matching_result(path, changed))

            changed_limit = result_metadata(
                "Method",
                "Scene",
                cache,
                [weight],
                {"radius": 0.1, "n_queries_limit": 10},
            )
            self.assertIsNone(matching_result(path, changed_limit))

    def test_content_or_checkpoint_change_invalidates_but_relocation_does_not(self):
        with tempfile.TemporaryDirectory() as root:
            cache, weight = self._inputs(root)
            digest_cache = os.path.join(root, "digests.json")
            with mock.patch.object(checkpoint_provenance, "_CACHE_FILE", digest_cache):
                original = result_metadata("Method", "Scene", cache, [weight])
                path = self._saved_result(root, original)

                relocated_cache = os.path.join(root, "moved-pool.json")
                relocated_weight = os.path.join(root, "moved-model.pth")
                shutil.copyfile(cache, relocated_cache)
                shutil.copyfile(weight, relocated_weight)
                relocated = result_metadata(
                    "Method", "Scene", relocated_cache, [relocated_weight]
                )
                self.assertIsNotNone(matching_result(path, relocated))

                old = os.stat(relocated_cache)
                with open(relocated_cache, "wb") as handle:
                    handle.write(b"BBBB")
                os.utime(relocated_cache, ns=(old.st_atime_ns, old.st_mtime_ns))
                changed_cache = result_metadata(
                    "Method", "Scene", relocated_cache, [relocated_weight]
                )
                self.assertIsNone(matching_result(path, changed_cache))

                old = os.stat(relocated_weight)
                with open(relocated_weight, "wb") as handle:
                    handle.write(b"WEIGHTS")
                os.utime(relocated_weight, ns=(old.st_atime_ns, old.st_mtime_ns))
                changed_weight = result_metadata(
                    "Method", "Scene", cache, [relocated_weight]
                )
                self.assertIsNone(matching_result(path, changed_weight))

    def test_malformed_payload_is_never_migrated(self):
        with tempfile.TemporaryDirectory() as root:
            cache, weight = self._inputs(root)
            metadata = result_metadata("Method", "Scene", cache, [weight])
            path = self._saved_result(root, metadata)
            with open(path, encoding="utf-8") as handle:
                result = json.load(handle)
            result["selected_ranks"].pop()
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(result, handle)

            self.assertIsNone(matching_result(path, metadata))


if __name__ == "__main__":
    unittest.main()
