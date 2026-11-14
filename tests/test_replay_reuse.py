import json
import os
import tempfile
import unittest
from unittest import mock

import run_paper_tables as tables


def _rows(queries=2, candidates=3, steps=None):
    steps = steps or tables.MAX_STEPS
    out = []
    for q in range(queries):
        for c in range(candidates):
            base = 10.0 * c + q
            out.append(
                {
                    "query": f"q{q}",
                    "candidate_name": f"cand{c}",
                    "rank": c + 1,
                    "error": base,
                    "step_errors": [base + i for i in range(steps)],
                    "inliers": [3 + c] * steps,
                    "scores": [0.1 * (c + 1)] * steps,
                    "area_step": [1.0] * steps,
                    "area_query": [1.0] * steps,
                    "scale_ratio": [0.5] * steps,
                }
            )
    return out


class RecordsCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        tables._RECORDS_CACHE.clear()
        self.addCleanup(tables._RECORDS_CACHE.clear)

    def path(self, name, rows):
        target = os.path.join(self.temp.name, name)
        with open(target, "w", encoding="utf-8") as handle:
            json.dump(rows, handle)
        return target

    def test_second_read_of_the_same_file_does_not_touch_disk(self):
        path = self.path("a.json", _rows())
        first = tables._grouped_records(path)
        with mock.patch("builtins.open", side_effect=AssertionError("re-read")):
            second = tables._grouped_records(path)
        self.assertIs(first, second)

    def test_distinct_files_stay_distinct(self):
        one = self.path("one.json", _rows(queries=2))
        two = self.path("two.json", _rows(queries=5))
        self.assertEqual(len(tables._grouped_records(one)), 2)
        self.assertEqual(len(tables._grouped_records(two)), 5)
        self.assertEqual(len(tables._grouped_records(one)), 2)

    def test_cache_is_bounded_and_evicts_least_recently_used(self):
        paths = [self.path(f"s{i}.json", _rows()) for i in range(5)]
        for path in paths:
            tables._grouped_records(path)
        self.assertLessEqual(len(tables._RECORDS_CACHE), tables._RECORDS_CACHE_SLOTS)
        self.assertNotIn(paths[0], tables._RECORDS_CACHE)
        self.assertIn(paths[-1], tables._RECORDS_CACHE)

    def test_widest_dataset_partition_rotation_still_hits(self):

        routes = [self.path(f"r{i}.json", _rows()) for i in range(3)]
        for _threshold in range(3):
            for route in routes:
                tables._grouped_records(route)
        self.assertEqual(len(tables._RECORDS_CACHE), 3)
        with mock.patch("builtins.open", side_effect=AssertionError("re-read")):
            for route in routes:
                tables._grouped_records(route)


class FeatureCacheKeyTests(unittest.TestCase):

    def test_channel_set_is_part_of_the_key(self):
        full = tables._variant_channels(tables.FULL_VARIANT)
        without_dino = tables._variant_channels(
            {**tables.FULL_VARIANT, "ABLATION_USE_DINO": "False"}
        )
        self.assertNotEqual(
            tuple(sorted(full.items())), tuple(sorted(without_dino.items()))
        )

    def test_random_window_variant_reads_a_different_file(self):

        from core.config import data_tag

        self.assertNotEqual(data_tag(random_window=True), data_tag(random_window=False))

    def test_cache_is_bounded(self):
        self.assertGreaterEqual(tables._FEATURES_CACHE_SLOTS, 3)
        self.assertLessEqual(tables._FEATURES_CACHE_SLOTS, 4)


if __name__ == "__main__":
    unittest.main()
