import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("", "pipeline", "replay"):
    path = os.path.join(_ROOT, sub) if sub else _ROOT
    if os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)
