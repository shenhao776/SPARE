import json
import os
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("DATASET", "business_campus")

from core.vpr_core import db_features_cache_valid
from sl_agent.cache_fingerprint import (
    build_fingerprint,
    settings_still_match,
)


class TestDbFeaturesGate(unittest.TestCase):
    def test_matching_set_is_valid(self):
        desc = np.zeros((3, 8))
        paths = ["/d/a.png", "/d/b.png", "/d/c.png"]
        self.assertTrue(db_features_cache_valid(desc, paths, list(reversed(paths))))

    def test_swapped_dataset_is_invalid(self):

        desc = np.zeros((3, 8))
        old = ["/d/a.png", "/d/b.png", "/d/c.png"]
        new = ["/d/a.png", "/d/b.png", "/d/X.png"]
        self.assertFalse(db_features_cache_valid(desc, old, new))

    def test_count_change_is_invalid(self):
        desc = np.zeros((3, 8))
        old = ["/d/a.png", "/d/b.png", "/d/c.png"]
        self.assertFalse(db_features_cache_valid(desc, old, old + ["/d/d.png"]))

    def test_internal_inconsistency_is_invalid(self):
        desc = np.zeros((2, 8))
        paths = ["/d/a.png", "/d/b.png", "/d/c.png"]
        self.assertFalse(db_features_cache_valid(desc, paths, paths))


class TestWalkTableGate(unittest.TestCase):
    def test_missing_pair_raises(self):

        import core.vpr_core as vc

        with tempfile.TemporaryDirectory() as d:
            scene_dir = os.path.join(d, "datasets", "FakeScene", "database", "rec")
            os.makedirs(scene_dir)
            for ts in (1000, 2000, 3000):
                open(os.path.join(scene_dir, f"{ts}.png"), "wb").close()
            open(os.path.join(scene_dir, "pose.txt"), "w").write(
                "1000 0 0 0 0 0 0 1\n2000 1 0 0 0 0 0 1\n3000 2 0 0 0 0 0 1\n"
            )

            class FakePM:
                def get_pose_info(self, p):
                    return {"global": [0.0, 0.0, 0.0]}

                def calculate_physical_distance(self, *_):
                    return 1.0

            vc._WALK_TABLES["FakeScene"] = ({"1000.png|2000.png": 0.1}, set())
            try:
                anchor = os.path.join(scene_dir, "2000.png")
                with self.assertRaises(KeyError):
                    vc.get_database_spatial_sequence(anchor, FakePM(), [2, 2])
            finally:
                vc._WALK_TABLES.pop("FakeScene", None)


class TestStateCacheIdentity(unittest.TestCase):
    def _stamp(self, n_queries, settings):
        return build_fingerprint([f"q{i}" for i in range(n_queries)], dict(settings))

    def test_query_count_change_forces_rebuild(self):
        base = {"top_k": 10, "step_budget": 0.5}
        old = self._stamp(100, base)
        self.assertFalse(settings_still_match(old, self._stamp(120, base)))

    def test_identical_inputs_reuse(self):
        base = {"top_k": 10, "step_budget": 0.5}
        self.assertTrue(
            settings_still_match(self._stamp(100, base), self._stamp(100, base))
        )


class TestBaselineResultBinding(unittest.TestCase):
    def test_pool_cache_content_change_invalidates(self):
        from tools.rerank_eval_common import result_metadata

        with tempfile.TemporaryDirectory() as d:
            cache = os.path.join(d, "test_trajectories.json")
            json.dump([{"q": 1}], open(cache, "w"))
            m_old = result_metadata("M", "S", cache, pool="NetVLAD_4096")
            json.dump([{"q": 2}], open(cache, "w"))
            m_new = result_metadata("M", "S", cache, pool="NetVLAD_4096")
            self.assertNotEqual(m_old["cache"]["sha256"], m_new["cache"]["sha256"])
            self.assertNotEqual(m_old, m_new)


if __name__ == "__main__":
    unittest.main()


class TestPoolCacheFreshness(unittest.TestCase):

    def _setup(self, d, cache_queries, view_queries):
        pool_dir = os.path.join(d, "assets", "NetVLAD_4096")
        os.makedirs(pool_dir)
        json.dump(
            [{"query": q, "rank": 1, "candidate_name": "c.png"} for q in cache_queries],
            open(os.path.join(pool_dir, "test_trajectories_S_no_roi.json"), "w"),
        )
        qdir = os.path.join(d, "datasets", "S", "queries_test")
        dbdir = os.path.join(d, "datasets", "S", "database")
        os.makedirs(qdir)
        os.makedirs(dbdir)
        for q in view_queries:
            open(os.path.join(qdir, f"{q}.png"), "wb").close()
        open(os.path.join(dbdir, "c.png"), "wb").close()

    def _list_images(self, folder):
        out = []
        for root, _, files in os.walk(folder):
            out += [os.path.join(root, f) for f in files if f.endswith(".png")]
        return out

    def test_matching_view_passes(self):
        from tools.rerank_eval_common import scene_inputs

        with tempfile.TemporaryDirectory() as d:
            self._setup(d, ["q1", "q2"], ["q1", "q2"])
            scene_inputs(d, "S", 0, self._list_images, pool="NetVLAD_4096")

    def test_swapped_data_raises(self):
        from tools.rerank_eval_common import scene_inputs

        with tempfile.TemporaryDirectory() as d:
            self._setup(d, ["q1", "q2"], ["q1", "qX"])
            with self.assertRaises(RuntimeError):
                scene_inputs(d, "S", 0, self._list_images, pool="NetVLAD_4096")
