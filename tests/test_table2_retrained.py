import contextlib
import sys
import unittest
from unittest import mock

import reproduce
import run_paper_tables as tables


class TableTwoRetrainedTests(unittest.TestCase):
    def _run_paper_main(self, *arguments):
        patched_names = (
            "phase_table1_draft",
            "phase_table1_thresholds",
            "phase_table2",
            "phase_table3_draft",
            "phase_table1_seeds",
            "phase_table3_thresholds",
            "phase_table3",
            "phase_table2_trainval",
            "phase_union_heads",
            "phase_union_mix",
            "_run_table_replay",
            "_require_runtime_dependencies",
            "init_run_log",
        )
        with contextlib.ExitStack() as stack:
            patched = {
                name: stack.enter_context(mock.patch.object(reproduce, name))
                for name in patched_names
            }
            stack.enter_context(
                mock.patch.object(sys, "argv", ["reproduce.py", *arguments])
            )
            reproduce.main()
            return {name: item.call_count for name, item in patched.items()}

    def test_retrained_table2_is_excluded_from_default_completion(self):
        default = reproduce._paper_completion_phases()
        enabled = reproduce._paper_completion_phases(True)

        self.assertNotIn(
            reproduce.phase_table2_trainval, [fn for _t, fn, _k in default]
        )
        self.assertIs(
            {t: fn for t, fn, _k in enabled}["cross-retriever train/val"],
            reproduce.phase_table2_trainval,
        )
        self.assertFalse(
            reproduce.PAPER_CONFIG["producers"]["table2"]["build_retrained_default"]
        )

    def test_union_heads_run_once_per_paper_run(self):

        many = self._run_paper_main("--paper-order", "NCLT", "Boreas", "CustomCampus")
        self.assertEqual(many["phase_union_heads"], 1)
        self.assertEqual(many["phase_union_mix"], 1)
        only = self._run_paper_main("--paper-order", "NCLT", "--only", "union")
        self.assertEqual(only["phase_union_heads"], 1)
        self.assertEqual(only["phase_union_mix"], 0)
        self.assertEqual(only["phase_table1_draft"], 0)

    def test_union_head_jobs_cover_ablation_rows_and_leave_one_out(self):
        jobs = reproduce._union_head_jobs()
        tags = [tag for _label, tag, _scenes, _radius, _overrides in jobs]
        rows = [
            row
            for row in reproduce.PAPER_CONFIG["ablation_rows"]
            if not row.get("default")
        ]
        self.assertEqual(tags.count("UnionMaps"), 1 + 3 + len(rows))
        members = set(reproduce.PAPER_CONFIG["default_head"]["members"])
        for label, tag, scenes, radius, overrides in jobs:
            self.assertTrue(set(scenes) <= members, label)
            if tag.startswith("LOO_"):
                self.assertLess(len(scenes), len(members))
                self.assertEqual(overrides, {})
        self.assertIn("LOO_4Seasons", tags)
        self.assertNotIn("LOO_Pitts30k", tags)

    def test_paper_cli_only_builds_retrained_table2_when_enabled(self):
        default = self._run_paper_main(
            "--paper-order", "NCLT", "--step-max-mode", "auto"
        )
        enabled = self._run_paper_main("--paper-order", "NCLT", "--table2-retrained")

        self.assertEqual(default["phase_table2_trainval"], 0)
        self.assertEqual(enabled["phase_table2_trainval"], 1)
        self.assertEqual(default["phase_table2"], 1)

    def test_scene_replaces_dataset_for_low_level_debugging(self):
        with (
            mock.patch.object(reproduce, "_require_runtime_dependencies"),
            mock.patch.object(reproduce, "init_run_log"),
            mock.patch.object(reproduce, "reproduce_trajectory") as trajectory,
            mock.patch.object(
                sys, "argv", ["reproduce.py", "--scene", "NCLT", "--stage", "build"]
            ),
        ):
            reproduce.main()
        trajectory.assert_called_once_with(
            "NCLT", ["build"], False, seeds=(str(reproduce.PAPER_SEEDS[0]),)
        )

        with (
            mock.patch.object(sys, "argv", ["reproduce.py", "--dataset", "NCLT"]),
            self.assertRaises(SystemExit) as rejected,
        ):
            reproduce.main()
        self.assertEqual(rejected.exception.code, 2)

    def test_renderer_routes_heads_to_selected_backbone_and_separate_title(self):
        calls = []

        def fake_metrics(*args, **kwargs):
            calls.append(kwargs["agent_backbone"])
            return None

        with (
            mock.patch.object(tables, "_safe_metrics", side_effect=fake_metrics),
            mock.patch.object(tables, "_print_table") as output,
        ):
            tables._table_backbones(["NCLT"], ["NetVLAD"], "unused.log", retrained=True)
        self.assertEqual(calls, ["NetVLAD"] * len(tables.SWEEP_THRESHOLDS))
        self.assertEqual(
            output.call_args.args[0],
            tables.PAPER_CONFIG["tables"]["backbones_retrained"],
        )

        calls.clear()
        with (
            mock.patch.object(tables, "_safe_metrics", side_effect=fake_metrics),
            mock.patch.object(tables, "_print_table") as output,
        ):
            tables._table_backbones(
                ["NCLT"], ["NetVLAD"], "unused.log", retrained=False
            )
        self.assertEqual(calls, ["MegaLoc"] * len(tables.SWEEP_THRESHOLDS))
        self.assertEqual(
            output.call_args.args[0], tables.PAPER_CONFIG["tables"]["backbones"]
        )

    def test_tail_phase_actually_trains_each_threshold_then_replays_retrained_csv(self):
        commands = []

        def fake_run(command, env_extra=None, required=True):
            commands.append((command, dict(env_extra or {})))
            return True

        with (
            mock.patch.object(
                reproduce, "TRAJECTORY_DATASETS", {"RobotCar": ["robotcar"]}
            ),
            mock.patch.object(reproduce, "SCENE_NAME", {"robotcar": "RobotCar"}),
            mock.patch.object(reproduce, "PAPER_BACKBONES", ("MegaLoc", "NetVLAD")),
            mock.patch.object(reproduce, "PAPER_THRESHOLDS", (10.0, 15.0)),
            mock.patch.object(reproduce, "PAPER_SEEDS", (2026,)),
            mock.patch.object(reproduce, "_build_backbone_test_caches"),
            mock.patch.object(reproduce, "_traj_env", return_value={}),
            mock.patch.object(reproduce, "run", side_effect=fake_run),
            mock.patch.object(reproduce, "_run_table_replay") as replay,
        ):
            failures = []
            reproduce.phase_table2_trainval("RobotCar", False, failures)

        train_calls = [item for item in commands if item[0][-1] == "train_agent.py"]
        self.assertEqual(len(train_calls), 3)
        self.assertEqual(
            [call[1].get("SUCCESS_THRESHOLD_TEST", "5") for call in train_calls],
            ["5", "10", "15"],
        )
        self.assertFalse(failures)
        replay.assert_called_once_with("RobotCar", tables="backbones_retrained")


if __name__ == "__main__":
    unittest.main()


class ReplayScheduleTests(unittest.TestCase):

    def _replay_calls(self, *arguments):
        patched_names = (
            "phase_table1_draft",
            "phase_table1_thresholds",
            "phase_table2",
            "phase_table3_draft",
            "phase_table1_seeds",
            "phase_table3_thresholds",
            "phase_table3",
            "phase_table2_trainval",
            "phase_union_heads",
            "phase_union_mix",
            "_require_runtime_dependencies",
            "init_run_log",
        )
        with contextlib.ExitStack() as stack:
            for name in patched_names:
                stack.enter_context(mock.patch.object(reproduce, name))
            replay = stack.enter_context(
                mock.patch.object(reproduce, "_run_table_replay")
            )
            stack.enter_context(
                mock.patch.object(sys, "argv", ["reproduce.py", *arguments])
            )
            reproduce.main()
            return replay.call_count

    def test_full_paper_run_replays_exactly_twice(self):
        self.assertEqual(self._replay_calls(), 2)

    def test_replay_count_does_not_grow_with_dataset_count(self):
        one = self._replay_calls("--paper-order", "NCLT")
        many = self._replay_calls("--paper-order", "NCLT", "Boreas", "CustomCampus")
        self.assertEqual(one, 2)
        self.assertEqual(many, 2)

    def test_draft_marker_is_printed_between_the_two_replays(self):
        with contextlib.ExitStack() as stack:
            for name in (
                "phase_table1_draft",
                "phase_table1_thresholds",
                "phase_table2",
                "phase_table3_draft",
                "phase_table1_seeds",
                "phase_table3_thresholds",
                "phase_table3",
                "phase_table2_trainval",
                "phase_union_heads",
                "phase_union_mix",
                "_require_runtime_dependencies",
                "init_run_log",
                "_run_table_replay",
            ):
                stack.enter_context(mock.patch.object(reproduce, name))
            stack.enter_context(
                mock.patch.object(
                    sys, "argv", ["reproduce.py", "--paper-order", "NCLT"]
                )
            )
            printed = stack.enter_context(mock.patch("builtins.print"))
            reproduce.main()

        messages = [str(call.args[0]) for call in printed.call_args_list if call.args]
        self.assertTrue(any("DRAFT_TABLES_READY" in m for m in messages))
        self.assertTrue(any("REPRODUCE_ALL_DONE" in m for m in messages))
