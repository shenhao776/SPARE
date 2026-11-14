import os
import tempfile
import unittest
from unittest import mock

import reproduce
import run_paper_tables
from core.baseline_protocols import BASELINE_PROTOCOLS


class PaperDatasetRegistryTests(unittest.TestCase):
    def test_refactor_preserves_frozen_paper_parameters(self):
        self.assertEqual(
            run_paper_tables.PAPER_DATASETS,
            [
                "RobotCar",
                "4Seasons",
                "NCLT",
                "CustomCampus",
                "Boreas",
                "Pitts30k",
                "MSLS",
            ],
        )
        self.assertEqual(
            run_paper_tables.PAPER_BACKBONES,
            [
                "NetVLAD",
                "EigenPlaces",
                "MixVPR",
                "MegaLoc",
                "SALAD",
                "CricaVPR",
                "BoQ",
                "SuperVLAD",
                "PairVPR",
            ],
        )
        self.assertEqual(run_paper_tables.SWEEP_THRESHOLDS, (None, 10.0, 15.0))
        self.assertEqual(reproduce.PAPER_SEEDS, (2026, 1, 2, 3, 4, 5))
        thresholds = dict(run_paper_tables.OWN_TEST_THRESHOLDS)
        flat = {
            k: v for k, v in thresholds.items() if k.lower() in ("pitts30k", "msls")
        }
        self.assertEqual(set(flat.values()), {25.0})
        self.assertEqual({v for k, v in thresholds.items() if k not in flat}, {5.0})
        self.assertEqual(reproduce.MODEL_ID, "MegaLoc_8448")

    def test_producer_and_renderer_register_the_same_paper_datasets(self):
        self.assertEqual(
            set(reproduce.PAPER_DATASETS),
            set(run_paper_tables.PAPER_DATASETS),
        )
        for dataset in reproduce.PAPER_DATASETS:
            self.assertIn(dataset, reproduce.TRAJECTORY_DATASETS)
            expected_scenes = [
                reproduce.SCENE_NAME[stem]
                for stem in reproduce.TRAJECTORY_DATASETS[dataset]
            ]
            self.assertEqual(
                expected_scenes,
                run_paper_tables.OWN_DATASETS[dataset],
            )

    def test_ablation_registries_are_kept_in_sync(self):
        self.assertEqual(
            set(reproduce.TABLE3_DATASETS),
            set(run_paper_tables.ABLATION_DATASETS),
        )

    def test_renderer_validates_schema_without_manual_evaluator_version(self):
        result = {
            "metadata": {
                "method": "Method",
                "scene": "RobotCar",
            },
            "n_queries": 1,
            "reranked_recall": {"5": 1.0, "10": 1.0, "15": 1.0},
        }
        self.assertTrue(
            run_paper_tables._valid_reranker_result(result, "Method", "RobotCar")
        )
        result["metadata"]["evaluator_version"] = -999
        self.assertTrue(
            run_paper_tables._valid_reranker_result(result, "Method", "RobotCar")
        )
        result["reranked_recall"].pop("15")
        self.assertFalse(
            run_paper_tables._valid_reranker_result(result, "Method", "RobotCar")
        )

    def test_patchnetvlad_protocol_version(self):
        self.assertNotIn("SeqNet (S1)", BASELINE_PROTOCOLS)
        self.assertEqual(BASELINE_PROTOCOLS["Patch-NetVLAD"], 2)

    def test_paper_order_can_select_a_unique_subset(self):
        self.assertEqual(
            reproduce._resolve_paper_order(
                ["Boreas", "CustomCampus", "NCLT", "4Seasons"]
            ),
            ["Boreas", "CustomCampus", "NCLT", "4Seasons"],
        )
        self.assertEqual(
            reproduce._resolve_paper_order(None),
            reproduce.PAPER_DATASETS,
        )
        with self.assertRaises(SystemExit):
            reproduce._resolve_paper_order(["Boreas", "Boreas"])
        with self.assertRaises(SystemExit):
            reproduce._resolve_paper_order(["NotADataset"])

    @mock.patch("reproduce._layout_params_changed", return_value=False)
    @mock.patch("core.config.resolve_dataset", return_value={})
    @mock.patch("reproduce._require_layout")
    @mock.patch("reproduce._layout_ready", return_value=True)
    @mock.patch("reproduce._require_portable_raw_input")
    @mock.patch("reproduce._prepare_portable_raw_input")
    @mock.patch("reproduce._traj_env", return_value={})
    @mock.patch("reproduce.run")
    def test_existing_layout_is_presence_cached(
        self,
        run,
        _traj_env,
        _prepare,
        _raw,
        layout_ready,
        require_layout,
        _resolve,
        _params_changed,
    ):
        reproduce.reproduce_trajectory("NCLT", ["layout"], False)
        layout_ready.assert_called_once_with("NCLT")
        require_layout.assert_called_once_with("NCLT", ran_layout=True)
        run.assert_not_called()

    def test_opted_in_step_budget_is_recalibrated_each_run(self):
        with tempfile.TemporaryDirectory() as root:
            config_dir = os.path.join(root, "configs", "datasets")
            os.makedirs(config_dir)
            config_path = os.path.join(config_dir, "robotcar.yaml")
            with open(config_path, "w", encoding="utf-8") as handle:
                handle.write(
                    "sequence:\n"
                    "  step_budget: 0.5\n"
                    "  step_budget_calibrated: true\n"
                )
            walk_path = os.path.join(root, "walk.json")
            with open(walk_path, "w", encoding="utf-8") as handle:
                handle.write('{"budget_half": 0.25}')
            with mock.patch.object(reproduce, "REPO", root):
                reproduce._write_calibrated_budget("RobotCar", "robotcar", walk_path)
            with open(config_path, encoding="utf-8") as handle:
                body = handle.read()
            self.assertIn("step_budget: 0.25", body)
            self.assertIn("step_budget_calibrated: true", body)


if __name__ == "__main__":
    unittest.main()
