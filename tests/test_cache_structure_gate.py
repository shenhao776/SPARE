import json
import os
import tempfile
import unittest
from unittest import mock

from sl_agent import cache_fingerprint as fp

SETTINGS = {"top_k": 10, "expected_steps": 5}


def _row(query, candidate, step_errors):
    return {
        "query": query,
        "candidate_name": candidate,
        "rank": int(candidate[-1]),
        "error": step_errors[len(step_errors) // 2],
        "step_errors": list(step_errors),
        "inliers": [5] * len(step_errors),
        "scores": [0.1] * len(step_errors),
        "area_step": [1.0] * len(step_errors),
        "area_query": [1.0] * len(step_errors),
        "scale_ratio": [0.5] * len(step_errors),
    }


def healthy_rows(queries=3, candidates=10, steps=5):
    rows = []
    for q in range(queries):
        for c in range(candidates):
            base = 10.0 * c + q
            rows.append(_row(f"q{q}", f"cand{c}", [base + i for i in range(steps)]))
    return rows


def cloned_rows(queries=3, candidates=10, steps=5):

    rows = []
    for q in range(queries):
        shared = [90.0 + i + q for i in range(steps)]
        for c in range(candidates):
            rows.append(_row(f"q{q}", f"cand{c}", shared))
    return rows


class CloneDetectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def write(self, name, rows):
        path = os.path.join(self.temp.name, name)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(rows, handle)
        return path

    def test_healthy_cache_passes(self):
        stats = fp.cache_stats(self.write("ok.json", healthy_rows()))
        self.assertEqual(stats["clone_groups"], 0)
        self.assertIsNone(fp.structural_error(stats, SETTINGS))

    def test_every_candidate_sharing_one_window_is_rejected(self):
        stats = fp.cache_stats(self.write("bad.json", cloned_rows()))
        self.assertEqual(stats["clone_groups"], 3)
        message = fp.structural_error(stats, SETTINGS)
        self.assertIsNotNone(message)
        self.assertIn("IDENTICAL", message)

    def test_a_single_cloned_group_is_enough_to_reject(self):

        rows = healthy_rows(queries=20)
        shared = [7.0, 8.0, 9.0, 10.0, 11.0]
        for row in rows:
            if row["query"] == "q4":
                row["step_errors"] = list(shared)
        stats = fp.cache_stats(self.write("one.json", rows))
        self.assertEqual(stats["clone_groups"], 1)
        self.assertIsNotNone(fp.structural_error(stats, SETTINGS))

    def test_single_candidate_queries_are_not_clones(self):
        rows = [_row("q0", "cand0", [1.0, 2.0, 3.0, 4.0, 5.0])]
        stats = fp.cache_stats(self.write("solo.json", rows))
        self.assertEqual(stats["clone_groups"], 0)
        self.assertIsNone(fp.structural_error(stats, SETTINGS))

    def test_duplicate_candidate_names_are_reported_as_duplication_not_cloning(self):

        rows = healthy_rows(queries=1)
        rows.append(dict(rows[0]))
        stats = fp.cache_stats(self.write("dupe.json", rows))
        self.assertEqual(stats["clone_groups"], 0)
        self.assertIn("rows per query", fp.structural_error(stats, SETTINGS) or "")


class ResumeGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.settings = {"top_k": 10, "expected_steps": 5, "split": "train"}
        self.fingerprint = fp.build_fingerprint(["a", "b", "c"], self.settings)

    def cache(self, name, rows):
        path = os.path.join(self.temp.name, name)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(rows, handle)
        return path

    def test_producer_stamp_lets_resume_skip_the_scan(self):
        path = self.cache("ok.json", healthy_rows())
        fp.write_fingerprint(path, self.fingerprint)
        with mock.patch.object(fp, "cache_stats") as scan:
            self.assertIsNone(fp.resume_reason(path, self.fingerprint))
        scan.assert_not_called()

    def test_unstamped_but_healthy_cache_is_scanned_once_then_stamped(self):
        path = self.cache("legacy.json", healthy_rows())
        stamp = {k: v for k, v in self.fingerprint.items() if k != "structure_schema"}
        fp.write_fingerprint(path, stamp)

        self.assertIsNone(fp.resume_reason(path, self.fingerprint))
        self.assertEqual(
            fp.read_fingerprint(path).get("structure_schema"), fp.STRUCTURE_SCHEMA
        )
        with mock.patch.object(fp, "cache_stats") as scan:
            self.assertIsNone(fp.resume_reason(path, self.fingerprint))
        scan.assert_not_called()

    def test_matching_key_does_not_rescue_a_cloned_cache(self):

        path = self.cache("clone.json", cloned_rows())
        stamp = {k: v for k, v in self.fingerprint.items() if k != "structure_schema"}
        fp.write_fingerprint(path, stamp)

        reason = fp.resume_reason(path, self.fingerprint)
        self.assertIsNotNone(reason)
        self.assertIn("IDENTICAL", reason)
        self.assertNotEqual(
            fp.read_fingerprint(path).get("structure_schema"), fp.STRUCTURE_SCHEMA
        )

    def test_migration_cannot_launder_a_cloned_cache(self):

        path = self.cache("migrate.json", cloned_rows())
        legacy = {
            "complete": True,
            "n_queries": self.fingerprint["n_queries"],
            "semantic_digest": "old-ast-digest",
            "settings": dict(self.settings),
        }
        with open(fp.meta_path(path), "w", encoding="utf-8") as handle:
            json.dump(legacy, handle)

        reason = fp.resume_reason(path, self.fingerprint)
        self.assertIsNotNone(reason)
        self.assertIn("IDENTICAL", reason)
        self.assertNotEqual(
            (fp.read_fingerprint(path) or {}).get("structure_schema"),
            fp.STRUCTURE_SCHEMA,
        )

    def test_migration_of_a_healthy_cache_still_resumes(self):
        path = self.cache("migrate_ok.json", healthy_rows())
        legacy = {
            "complete": True,
            "n_queries": self.fingerprint["n_queries"],
            "semantic_digest": "old-ast-digest",
            "settings": dict(self.settings),
        }
        with open(fp.meta_path(path), "w", encoding="utf-8") as handle:
            json.dump(legacy, handle)

        self.assertIsNone(fp.resume_reason(path, self.fingerprint))
        self.assertEqual(
            fp.read_fingerprint(path).get("structure_schema"), fp.STRUCTURE_SCHEMA
        )


if __name__ == "__main__":
    unittest.main()


class KeyIsResultAffectingOnlyTests(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def _cache(self, settings, name="c.json"):
        path = os.path.join(self.temp.name, name)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(healthy_rows(), handle)
        fp.write_fingerprint(path, fp.build_fingerprint(["a", "b"], settings))
        return path

    def test_signature_keeps_only_weights_and_preprocessing(self):
        self.assertEqual(set(fp._RETRIEVER_KEY_FIELDS), {"transform", "weight_files"})

    def test_plumbing_change_does_not_rebuild(self):

        weights = {"megaloc/model.safetensors": "sha-unchanged"}
        wide = {
            "top_k": 10,
            "expected_steps": 5,
            "retriever": {
                "selector": "megaloc",
                "model_id": "MegaLoc_8448",
                "loader": "torch_hub",
                "backbone": None,
                "output_dim": None,
                "hub_repo": "gmberton/MegaLoc:main",
                "hub_entry": "get_trained_model",
                "hub_kwargs": None,
                "transform": {"type": "multiple_of_14", "short_edge": 322},
                "weight_files": weights,
            },
        }
        path = self._cache(wide)

        narrow = {
            "top_k": 10,
            "expected_steps": 5,
            "retriever": {
                "transform": {"type": "multiple_of_14", "short_edge": 322},
                "weight_files": weights,
            },
        }
        self.assertIsNone(
            fp.resume_reason(path, fp.build_fingerprint(["a", "b"], narrow))
        )

    def test_new_weights_still_rebuild(self):

        base = {
            "top_k": 10,
            "expected_steps": 5,
            "retriever": {
                "transform": {"type": "multiple_of_14", "short_edge": 322},
                "weight_files": {"megaloc/model.safetensors": "sha-A"},
            },
        }
        path = self._cache(base)
        changed = json.loads(json.dumps(base))
        changed["retriever"]["weight_files"]["megaloc/model.safetensors"] = "sha-B"

        reason = fp.resume_reason(path, fp.build_fingerprint(["a", "b"], changed))
        self.assertIsNotNone(reason)

    def test_transform_change_still_rebuilds(self):
        base = {
            "top_k": 10,
            "expected_steps": 5,
            "retriever": {
                "transform": {"type": "multiple_of_14", "short_edge": 322},
                "weight_files": {"m": "sha"},
            },
        }
        path = self._cache(base)
        changed = json.loads(json.dumps(base))
        changed["retriever"]["transform"]["short_edge"] = 448

        self.assertIsNotNone(
            fp.resume_reason(path, fp.build_fingerprint(["a", "b"], changed))
        )

    def test_protocol_numbers_still_rebuild(self):
        base = {"top_k": 10, "expected_steps": 5, "step_budget": 0.5}
        path = self._cache(base)
        self.assertIsNotNone(
            fp.resume_reason(
                path,
                fp.build_fingerprint(["a", "b"], {**base, "step_budget": 0.6}),
            )
        )

    def test_key_contains_no_code_or_commit_identity(self):

        import core.config

        settings = {"top_k": 10, "retriever": {"weight_files": {"m": "sha"}}}
        blob = json.dumps(fp.build_fingerprint(["a"], settings)).lower()
        for forbidden in ("ast", "commit", "git", "mtime", "source_sha", "producer"):
            self.assertNotIn(forbidden, blob)


class NoManualVersionTests(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def cache(self, rows, name="c.json"):
        path = os.path.join(self.temp.name, name)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(rows, handle)
        return path

    def test_no_manual_version_remains_in_the_key(self):
        stamp = fp.build_fingerprint(["a"], {"top_k": 10})
        self.assertNotIn("version", stamp)
        self.assertNotIn("content_version", json.dumps(stamp))
        self.assertFalse(hasattr(fp, "cache_version"))

    def test_short_channel_is_detected_as_stale(self):

        rows = healthy_rows(steps=5)
        for row in rows:
            row["dino_sim"] = [0.7]
        path = self.cache(rows)
        why = fp.injected_shape_error(path, "dino", 5)
        self.assertIsNotNone(why)
        self.assertIn("1 steps, expected 5", why)

    def test_full_length_channel_is_accepted(self):
        rows = healthy_rows(steps=5)
        for row in rows:
            row["dino_sim"] = [0.1, 0.2, 0.3, 0.4, 0.5]
        self.assertIsNone(fp.injected_shape_error(self.cache(rows), "dino", 5))

    def test_missing_channel_is_stale(self):
        rows = healthy_rows(steps=5)
        self.assertIn("missing", fp.injected_shape_error(self.cache(rows), "dino", 5))

    def test_every_injector_declares_the_fields_it_owns(self):
        for name in ("dino", "step_meta", "zoom"):
            self.assertTrue(fp.INJECTOR_FIELDS.get(name), name)

    def test_step_meta_needs_all_four_of_its_fields(self):
        rows = healthy_rows(steps=5)
        for row in rows:
            row["step_offset_m"] = [0.0] * 5
            row["step_turn_deg"] = [0.0] * 5
            row["step_change"] = [0.0] * 5
        why = fp.injected_shape_error(self.cache(rows), "step_meta", 5)
        self.assertIn("step_valid", why)

    def test_injector_reruns_when_shape_is_wrong_even_if_inputs_match(self):
        rows = healthy_rows(steps=5)
        for row in rows:
            row["dino_sim"] = [0.7]
        path = self.cache(rows)
        provenance = {"model_name": "dinov2_vitb14", "weight_sha256": "abc"}
        fp.write_fingerprint(path, fp.build_fingerprint(["a"], {"top_k": 10}))
        fp.refresh_output_fingerprint(path, dino=provenance)

        self.assertFalse(
            fp.injector_current(path, "dino", provenance, expected_steps=5)
        )

    def test_injector_skips_when_inputs_and_shape_both_agree(self):
        rows = healthy_rows(steps=5)
        for row in rows:
            row["dino_sim"] = [0.1, 0.2, 0.3, 0.4, 0.5]
        path = self.cache(rows)
        provenance = {"model_name": "dinov2_vitb14", "weight_sha256": "abc"}
        fp.write_fingerprint(path, fp.build_fingerprint(["a"], {"top_k": 10}))
        fp.refresh_output_fingerprint(path, dino=provenance)

        self.assertTrue(fp.injector_current(path, "dino", provenance, expected_steps=5))

    def test_changed_learned_inputs_still_rerun(self):
        rows = healthy_rows(steps=5)
        for row in rows:
            row["dino_sim"] = [0.1, 0.2, 0.3, 0.4, 0.5]
        path = self.cache(rows)
        fp.write_fingerprint(path, fp.build_fingerprint(["a"], {"top_k": 10}))
        fp.refresh_output_fingerprint(path, dino={"weight_sha256": "old"})

        self.assertFalse(
            fp.injector_current(
                path, "dino", {"weight_sha256": "new"}, expected_steps=5
            )
        )

    def test_legacy_stamp_with_manual_version_is_ignored_not_rejected(self):

        rows = healthy_rows(steps=5)
        for row in rows:
            row["dino_sim"] = [0.1, 0.2, 0.3, 0.4, 0.5]
        path = self.cache(rows)
        fp.write_fingerprint(path, fp.build_fingerprint(["a"], {"top_k": 10}))
        fp.refresh_output_fingerprint(
            path, dino={"content_version": 1, "weight_sha256": "abc"}
        )

        self.assertTrue(
            fp.injector_current(
                path, "dino", {"weight_sha256": "abc"}, expected_steps=5
            )
        )
