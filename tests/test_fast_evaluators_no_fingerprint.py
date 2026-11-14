import inspect
import unittest

import run_paper_tables
from tools import fast_offline_eval


class FastEvaluatorsHaveNoFingerprintTests(unittest.TestCase):
    def test_fast_offline_eval_reads_files_directly(self):
        source = inspect.getsource(fast_offline_eval.main)
        self.assertNotIn("training_fingerprint", source)
        self.assertNotIn("output_sha256", source)
        self.assertNotIn(".meta.json", source)

    def test_table_replay_reads_files_directly(self):
        agent = inspect.getsource(run_paper_tables._own_agent)
        partition = inspect.getsource(run_paper_tables._run_own_partition)
        baselines = inspect.getsource(run_paper_tables._baseline_values)
        for source in (agent, partition, baselines):
            self.assertNotIn("fingerprint", source)
            self.assertNotIn("output_sha256", source)
            self.assertNotIn(".meta.json", source)


if __name__ == "__main__":
    unittest.main()
