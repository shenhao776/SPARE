import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.environ.setdefault("DATASET", "robotcar")


def perturb(value):

    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float)):
        return value + 999
    if isinstance(value, str):
        return value + "__perturbed"
    if isinstance(value, (list, tuple)):
        return list(value) + [999]
    if value is None:
        return "__was_none"
    return f"__perturbed:{value!r}"
