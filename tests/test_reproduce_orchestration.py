import os
import re
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("DATASET", "nclt")

import reproduce as R


class TestPaperCoverage(unittest.TestCase):

    def test_不带参数等于全部论文数据集(self):

        self.assertEqual(R._resolve_paper_order(None), list(R.PAPER_DATASETS))
        self.assertEqual(len(R.PAPER_DATASETS), 7)

    def test_paper_order_拒绝未知与重复(self):

        for bad in (["NoSuchDataset"], ["NCLT", "NCLT"], ["NCLT", "nclt"]):
            with self.subTest(bad=bad):
                with self.assertRaises(SystemExit):
                    R._resolve_paper_order(bad)

    def test_空参数与不传参数同义(self):
        self.assertEqual(R._resolve_paper_order([]), list(R.PAPER_DATASETS))

    def test_DRAFT四阶段的内容(self):

        src = open(os.path.join(ROOT, "reproduce.py"), encoding="utf-8").read()
        block = re.search(r"DRAFT = \((.*?)\n        \)", src, re.S)
        self.assertIsNotNone(block, "找不到 DRAFT 定义")
        fns = re.findall(r"phase_\w+", block.group(1))
        self.assertEqual(
            fns,
            [
                "phase_table1_draft",
                "phase_table1_thresholds",
                "phase_table2",
                "phase_table3_draft",
            ],
        )

    def test_补全阶段包含六种子且不再有SeqNet(self):
        names = [fn.__name__ for _title, fn, _key in R._paper_completion_phases(False)]
        for need in ("phase_table1_seeds", "phase_table3"):
            self.assertIn(need, names)
        self.assertFalse(hasattr(R, "phase_seqnet"))
        self.assertNotIn("phase_table2_trainval", names)
        self.assertIn(
            "phase_table2_trainval",
            [fn.__name__ for _t, fn, _key in R._paper_completion_phases(True)],
        )

    def test_表IV不在无参数流程里(self):

        src = open(os.path.join(ROOT, "reproduce.py"), encoding="utf-8").read()
        draft = re.search(r"DRAFT = \((.*?)\n        \)", src, re.S).group(1)
        self.assertNotIn("crop", draft.lower())
        completion = [t for t, _fn, _key in R._paper_completion_phases(True)]
        self.assertFalse([t for t in completion if "crop" in t.lower()])

    def test_消融数据集是论文数据集的子集(self):

        self.assertTrue(R.TABLE3_DATASETS <= set(R.PAPER_DATASETS))
        self.assertTrue(R.TABLE3_DATASETS, "消融数据集为空")

    def test_消融行清单与训练臂自洽(self):

        trained = {label for label, _ in R._paper_variants()}
        default = {
            r["label"] for r in R.PAPER_CONFIG["ablation_rows"] if r.get("default")
        }
        untrained = {
            r["label"]
            for r in R.PAPER_CONFIG["ablation_rows"]
            if r.get("variant") is None
        }
        raw_row = {"Raw VPR (MegaLoc, no re-ranking)"}
        covered = trained | default | untrained | raw_row
        for _short, full in R.PAPER_CONFIG["paper_csv"]["ablation_variant_labels"]:
            self.assertIn(full, covered, f"消融行 {full!r} 没有任何产出路径")

    def test_每个论文数据集都有场景映射(self):
        for ds in R.PAPER_DATASETS:
            stems = R.TRAJECTORY_DATASETS.get(ds)
            self.assertTrue(stems, f"{ds} 没有场景映射,阶段会静默跑空")
            for stem in stems:
                self.assertIn(stem, R.SCENE_NAME, f"{stem} 没有场景名")


def _stub_runs(fn, *args):

    real_run, real_train = R.run, R._train_one
    calls = []

    def fake_run(cmd, env_extra=None, required=True):
        calls.append((list(cmd), dict(env_extra or {})))
        return True

    def fake_train(stem, label, variant, seed, force):
        env = {**R._traj_env(stem), **(variant or {}), "TRAIN_SEED": str(seed)}
        calls.append(([R.PY, "train_agent.py"], env))
        return True, []

    R.run, R._train_one = fake_run, fake_train
    try:
        fn(*args)
    finally:
        R.run, R._train_one = real_run, real_train
    return calls


class TestPhaseDispatch(unittest.TestCase):

    def _dispatch(self, fn, dataset, *extra):
        try:
            return _stub_runs(fn, dataset, False, [], *extra)
        except SystemExit as exc:
            self.skipTest(f"{dataset} 资产不全,跳过编排检查: {str(exc)[:60]}")

    def test_打桩确实拦住了所有执行出口(self):

        import subprocess

        class _Blocked:

            returncode = 0
            stdout = ""

        real_run, real_popen = subprocess.run, subprocess.Popen
        escaped = []

        def trap(*args, **kwargs):
            escaped.append(args[0] if args else kwargs.get("args"))
            return _Blocked()

        subprocess.run, subprocess.Popen = trap, trap
        try:
            ds = sorted(R.TABLE3_DATASETS)[0]
            self._dispatch(R.phase_table3_thresholds, ds)
        finally:
            subprocess.run, subprocess.Popen = real_run, real_popen
        self.assertEqual(
            escaped, [], f"有 {len(escaped)} 条命令逃过打桩真的执行了: {escaped[:2]}"
        )

    def _run_train_one_with(self, returncode, stdout):

        import subprocess

        class _Result:
            def __init__(self):
                self.returncode = returncode
                self.stdout = stdout

        calls = []
        real_run, real_sleep = subprocess.run, R.time.sleep
        subprocess.run = lambda *a, **k: (calls.append(1), _Result())[1]
        R.time.sleep = lambda _s: None
        try:
            ok, _lines = R._train_one("nclt", "x", {}, 2026, False)
        finally:
            subprocess.run, R.time.sleep = real_run, real_sleep
        return ok, len(calls)

    def test_训练遇到瞬时GPU故障会重试(self):

        ok, n = self._run_train_one_with(1, f"boom {R.TRANSIENT_GPU_MARKERS[0]} boom")
        self.assertFalse(ok)
        self.assertEqual(
            n, R.TRANSIENT_RETRIES + 1, "瞬时 GPU 故障没有按 run() 的次数重试"
        )

    def test_普通失败不重试(self):

        ok, n = self._run_train_one_with(1, "ValueError: channel 'dino_sim' missing")
        self.assertFalse(ok)
        self.assertEqual(n, 1)

    def test_成功不重试(self):
        ok, n = self._run_train_one_with(0, "Training finished")
        self.assertTrue(ok)
        self.assertEqual(n, 1)

    def test_消融阶段只在消融数据集上派活(self):

        for ds in R.PAPER_DATASETS:
            with self.subTest(ds=ds):
                calls = self._dispatch(R.phase_table3_thresholds, ds)
                if ds in R.TABLE3_DATASETS:
                    self.assertTrue(calls, f"{ds} 在消融名单里却没派活")
                else:
                    self.assertEqual(calls, [], f"{ds} 不在消融名单里却派了活")

    def test_派出的训练命令都带场景与种子(self):

        ds = sorted(R.TABLE3_DATASETS)[0]
        calls = self._dispatch(R.phase_table3_thresholds, ds)
        trains = [(c, e) for c, e in calls if c[-1].endswith("train_agent.py")]
        self.assertTrue(trains, "消融阶段没派出任何训练")
        for cmd, env in trains:
            self.assertIn("DATASET", env)
            self.assertIn("TRAIN_SEED", env)

    def test_阈值扫描派出的是不同阈值(self):

        ds = sorted(R.TABLE3_DATASETS)[0]
        calls = self._dispatch(R.phase_table3_thresholds, ds)
        ths = {
            e.get("SUCCESS_THRESHOLD_TEST")
            for c, e in calls
            if c[-1].endswith("train_agent.py")
        }
        self.assertGreaterEqual(len(ths), 2, f"阈值没有区分度: {ths}")


if __name__ == "__main__":
    unittest.main()
