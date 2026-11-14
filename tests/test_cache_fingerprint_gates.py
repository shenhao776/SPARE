import unittest

from tests.fingerprint_helpers import perturb as _perturb

from sl_agent.cache_fingerprint import (
    build_fingerprint,
    settings_still_match,
    state_build_settings,
)

QUERIES = [f"q{i}.png" for i in range(100)]


def _fp(queries, settings):
    return build_fingerprint(queries, settings)


class TestFingerprintIdentity(unittest.TestCase):
    def test_identical_inputs_reuse(self):
        base = {"top_k": 10, "step_budget": 0.5}
        self.assertEqual(
            _fp(QUERIES, base)["cache_key"], _fp(QUERIES, base)["cache_key"]
        )
        self.assertTrue(settings_still_match(base, dict(base)))

    def test_query_count_change_invalidates(self):
        base = {"top_k": 10}
        self.assertNotEqual(
            _fp(QUERIES, base)["cache_key"],
            _fp(QUERIES + ["extra.png"], base)["cache_key"],
        )

    def test_每个影响数值的设置改了都必须失效(self):

        base = state_build_settings("test")
        self.assertTrue(base, "state_build_settings 返回空,指纹形同虚设")
        for key, value in sorted(base.items()):
            with self.subTest(key=key):
                changed = dict(base)
                changed[key] = _perturb(value)
                self.assertNotEqual(
                    changed[key], value, f"{key} 的扰动值与原值相同,这条断言无效"
                )
                self.assertFalse(
                    settings_still_match(base, changed),
                    f"改了 {key} 却仍判定缓存有效 —— 它影响数值就必须进指纹;"
                    f"若它确实不影响数值,把它移出 state_build_settings()",
                )
                self.assertNotEqual(
                    _fp(QUERIES, base)["cache_key"], _fp(QUERIES, changed)["cache_key"]
                )

    def test_遍历用的扰动函数对每种取值都真的改了值(self):

        for value in (0, 1, -1, 0.0, 2.5, True, False, "", "x", [], [1, 2], None, (3,)):
            with self.subTest(value=value):
                self.assertNotEqual(_perturb(value), value)

    def test_retriever_收窄字段不会误伤旧缓存(self):

        old = {
            "retriever": {
                "name": "MegaLoc",
                "dim": 8448,
                "loader": "hub",
                "alias": "megaloc",
            }
        }
        new = {"retriever": {"name": "MegaLoc", "dim": 8448}}
        self.assertTrue(settings_still_match(old, new))
        self.assertFalse(
            settings_still_match(old, {"retriever": {"name": "SALAD", "dim": 8448}})
        )

    def test_state_build_settings_覆盖协议关键量(self):

        base = state_build_settings("test")
        for key in (
            "model_id",
            "top_k",
            "spatial_steps",
            "step_budget",
            "step_feature_tau",
            "step_max_dist",
            "geometry_version",
            "random_window",
            "retriever",
        ):
            self.assertIn(key, base, f"协议关键量 {key} 不在缓存指纹里")

    def test_已知盲区_图像内容不进指纹(self):

        base = state_build_settings("test")
        self.assertNotIn("layout_digest", base)
        self.assertNotIn("crop", base)
        self.assertEqual(
            _fp(QUERIES, base)["cache_key"], _fp(list(QUERIES), base)["cache_key"]
        )


if __name__ == "__main__":
    unittest.main()
