import ast
import os
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILDERS = [
    "sl_agent/build_sl_dataset_train.py",
    "sl_agent/build_sl_dataset_val.py",
    "sl_agent/build_sl_dataset_test.py",
]


def _call_name(node):
    return getattr(node.func, "id", getattr(node.func, "attr", ""))


def _candidate_loops(tree):

    for node in ast.walk(tree):
        if not isinstance(node, ast.For) or not node.body:
            continue
        if ast.unparse(node.body[0]).startswith("rel_p = db_rel_paths"):
            yield node


def _uses_spatial_sequence(loop):
    return any(
        isinstance(n, ast.Call) and _call_name(n) == "get_database_spatial_sequence"
        for n in ast.walk(loop)
    )


def _rebinds_before_use(loop):
    for stmt in loop.body:
        for n in ast.walk(stmt):
            if isinstance(n, ast.Assign) and any(
                getattr(t, "id", "") == "db_abs_path" for t in n.targets
            ):
                return True
            if isinstance(n, ast.Call) and (
                _call_name(n) == "get_database_spatial_sequence"
            ):
                return False
    return False


class TestCandidateLoopAnchor(unittest.TestCase):
    def test_candidate_loop_rebinds_db_abs_path(self):
        for rel in BUILDERS:
            with self.subTest(builder=rel):
                path = os.path.join(ROOT, rel)
                tree = ast.parse(open(path, encoding="utf-8").read())
                loops = [l for l in _candidate_loops(tree) if _uses_spatial_sequence(l)]
                self.assertTrue(
                    loops,
                    f"{rel}: candidate loop calling "
                    f"get_database_spatial_sequence not found",
                )
                for loop in loops:
                    self.assertTrue(
                        _rebinds_before_use(loop),
                        f"{rel}: candidate loop uses get_database_spatial_sequence "
                        f"without first rebinding db_abs_path from its own rel_p -- "
                        f"every candidate would inherit the previous loop's stale "
                        f"anchor (clone bug)",
                    )


if __name__ == "__main__":
    unittest.main()
