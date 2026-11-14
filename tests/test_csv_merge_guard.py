import os
import re
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("DATASET", "nclt")

import run_paper_tables as RPT


class TestMissingPredicate(unittest.TestCase):
    def test_三段式缺失必须判为缺失(self):

        for value in ("--/--/--", "--", "-", "", "  --/--/--  "):
            with self.subTest(value=value):
                self.assertTrue(RPT._is_missing(value), f"{value!r} 应判为缺失")

    def test_有数的格不能判为缺失(self):
        for value in ("3.9/9.4/13.5", "41.6±1.6", "0.0/0.0/0.0", "28.5"):
            with self.subTest(value=value):
                self.assertFalse(RPT._is_missing(value), f"{value!r} 不该判为缺失")

    def test_部分有数不算缺失(self):

        self.assertFalse(RPT._is_missing("--/9.4/--"))
        self.assertFalse(RPT._is_missing("3.9/--/--"))

    def test_零不是缺失(self):

        self.assertFalse(RPT._is_missing("0.0/0.0/0.0"))
        self.assertFalse(RPT._is_missing(0.0))


class TestWriteMergedUsesPredicate(unittest.TestCase):

    def _source(self):
        return open(os.path.join(ROOT, "run_paper_tables.py"), encoding="utf-8").read()

    def test_保护走_is_missing而不是裸的NA集合(self):
        src = self._source()
        block = re.search(
            r"def _write_merged\(.*?\n        with open\(path, \"w\"", src, re.S
        )
        self.assertIsNotNone(block, "找不到 _write_merged 的合并段")
        body = block.group(0)
        self.assertIn("_is_missing", body, "保护没用 _is_missing —— 三段式会被漏判")
        self.assertNotIn(
            "in _NA_TOKENS |",
            body,
            "又退回裸的 _NA_TOKENS 判据了,那认不出 '--/--/--'",
        )


class TestPartialRenderDoesNotCrash(unittest.TestCase):

    def _source(self):
        return open(os.path.join(ROOT, "run_paper_tables.py"), encoding="utf-8").read()

    def test_按数据集取格一律用get(self):

        src = self._source()
        for bad in (r"_cell3\(got\[d\]\)", r"_cell3\(per\[\(b, k\)\]\[d\]\)"):
            with self.subTest(pattern=bad):
                self.assertFalse(
                    re.search(bad, src),
                    f"发现裸索引 {bad} —— 缺席的数据集会让整个重放 KeyError",
                )

    def test_消融CSV的列取自配置而非渲染结果(self):

        src = self._source()
        self.assertIn("abl_ds = list(ABLATION_DATASETS)", src)


if __name__ == "__main__":
    unittest.main()
