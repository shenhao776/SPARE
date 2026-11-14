import os
import subprocess
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestModelFlag(unittest.TestCase):
    def test_help_里有_model_参数(self):
        out = subprocess.run(
            [
                sys.executable,
                (
                    "pipeline/reproduce.py"
                    if os.path.exists("pipeline/reproduce.py")
                    else "reproduce.py"
                ),
                "--help",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=180,
        )
        self.assertEqual(out.returncode, 0)
        self.assertIn("--model", out.stdout)

    def test_环境变量传骨干会被剥掉(self):

        env = {
            **os.environ,
            "MODEL": "boq",
            "VPR_MODEL_TYPE": "BoQ",
            "DATASET": "robotcar",
        }
        out = subprocess.run(
            [
                sys.executable,
                "-c",
                "import core.env_isolation as e;"
                "print(sorted(e.scrub_environment().keys()))",
            ],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(out.returncode, 0, out.stderr[-300:])
        self.assertIn("MODEL", out.stdout)
        self.assertIn("VPR_MODEL_TYPE", out.stdout)

    def test_未知骨干当场报错(self):
        out = subprocess.run(
            [
                sys.executable,
                (
                    "pipeline/reproduce.py"
                    if os.path.exists("pipeline/reproduce.py")
                    else "reproduce.py"
                ),
                "--model",
                "NoSuchBackbone",
                "--scene",
                "RobotCar",
                "--stage",
                "layout",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=180,
        )
        self.assertNotEqual(out.returncode, 0)
        self.assertIn("不认识", out.stdout + out.stderr)


if __name__ == "__main__":
    unittest.main()
