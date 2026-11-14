import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("DATASET", "nclt")

import reproduce as R

KEPT = {
    "Ours (single-frame, default)",
    "w/o DINOv2 appearance",
    "w/o VPR-tuned appearance (MegaLoc patches)",
    "VPR-tuned appearance shuffled (negative control)",
    "w/o magnification (zoom)",
    "w/o patch appearance (match geometry + magnification)",
    "w/o match geometry (patch appearance + magnification)",
    "Linear head (same seven inputs)",
}


class TestAblationScope(unittest.TestCase):
    def test_保留的行一个不少(self):
        labels = {row.get("label") for row in R.PAPER_CONFIG["ablation_rows"]}
        self.assertEqual(labels, KEPT, "消融表的行清单与预期不符")

    def test_出表清单与行清单一致(self):

        csv_labels = {
            full
            for _short, full in R.PAPER_CONFIG["paper_csv"]["ablation_variant_labels"]
        }
        row_labels = {row.get("label") for row in R.PAPER_CONFIG["ablation_rows"]}
        self.assertEqual(csv_labels - {"Raw VPR (MegaLoc, no re-ranking)"}, row_labels)

    def test_训练臂不含默认行(self):
        trained = {label for label, _ in R._paper_variants()}
        self.assertNotIn("Ours (single-frame, default)", trained)
        self.assertEqual(len(trained), len(KEPT) - 1)

    def test_默认协议是单帧七维无输入归一化(self):
        from core.config import agent_input_dim, agent_tag, paper_protocol_env

        env = paper_protocol_env()
        self.assertEqual(env["ABLATION_ARCH"], "single_frame")
        self.assertEqual(env["ABLATION_INPUT_NORM"], "False")
        for key in (
            "ABLATION_USE_DERIVATIVES",
            "ABLATION_USE_PROGRESS",
            "ABLATION_USE_POSE",
        ):
            self.assertEqual(env[key], "False", key)
        self.assertEqual(env["ABLATION_USE_VPRPATCH"], "True")
        self.assertEqual(env["ABLATION_USE_DINO"], "True")
        self.assertEqual(agent_input_dim(), 7)
        self.assertIn("arch=single_frame", agent_tag())
        self.assertTrue(agent_tag().endswith("_innorm=False"))

    def test_消融数据集与论文表6一致(self):
        self.assertEqual(R.TABLE3_DATASETS, {"CustomCampus", "Pitts30k", "MSLS"})


class TestTrainWorkers(unittest.TestCase):
    def test_并行度至少为一且不超过配置(self):
        configured = int(R.PAPER_CONFIG["producers"]["train_workers"])
        workers = R._train_workers()
        self.assertGreaterEqual(workers, 1)
        self.assertLessEqual(
            workers,
            configured,
            "并行度不得超过配置值 —— 每路 2.5 GB 常驻,超了会把机器打进 swap,"
            "而 swap 之后训练比串行还慢且不报错",
        )


if __name__ == "__main__":
    unittest.main()
