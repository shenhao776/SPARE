import json
import os
import tempfile
import unittest
from unittest import mock

from tools import prepare_robotcar as robotcar


class PrepareRobotCarTests(unittest.TestCase):
    def _portable_output(self, root, complete=True):
        counts = {}
        for role, output_name in robotcar.ROLES.items():
            image_dir = os.path.join(root, output_name, "images")
            os.makedirs(image_dir, exist_ok=True)
            if complete:
                with open(os.path.join(image_dir, "1.jpg"), "wb") as handle:
                    handle.write(b"jpeg")
            with open(os.path.join(root, output_name, "pose.txt"), "w") as handle:
                handle.write("timestamp tx ty tz qx qy qz qw\n")
            counts[role] = {"selected": 1}
        parameters = {
            "sessions": {role: f"session-{role}" for role in robotcar.ROLES},
            "spacing_m": 1.5,
            "angle_deg": 10.0,
            "max_pose_delta_ms": 100.0,
            "bayer_pattern": "gbrg",
            "jpeg_quality": 95,
        }
        manifest = {
            "format_version": 1,
            "parameters": parameters,
            "counts": counts,
        }
        with open(os.path.join(root, robotcar.MANIFEST_NAME), "w") as handle:
            json.dump(manifest, handle)
        return parameters

    def test_complete_tree_is_reused_without_touching_the_source(self):
        with tempfile.TemporaryDirectory() as temp:
            source = os.path.join(temp, "source")
            output = os.path.join(temp, "output")
            os.makedirs(source)
            os.makedirs(output)
            parameters = self._portable_output(output)
            with mock.patch.object(robotcar, "_build_role") as build:
                robotcar.prepare_dataset(source, output, dict(parameters))
            build.assert_not_called()

    def test_portable_output_does_not_require_raw_source(self):
        with tempfile.TemporaryDirectory() as output:
            parameters = self._portable_output(output)
            with mock.patch.object(robotcar, "_build_role") as build:
                robotcar.prepare_dataset(None, output, dict(parameters))
            build.assert_not_called()

    def test_no_manual_version_or_fingerprint_remains(self):
        for name in (
            "PREPARE_CONTENT_VERSION",
            "SOURCE_FINGERPRINT_SCHEMA",
            "LEGACY_V3_PRODUCER_SHA256",
            "_source_fingerprint",
            "_source_fingerprint_v2",
            "_source_fingerprint_v3",
            "_legacy_source_fingerprint",
        ):
            self.assertFalse(hasattr(robotcar, name), name)

    def test_stale_manifest_contents_cannot_force_a_rebuild(self):

        with tempfile.TemporaryDirectory() as temp:
            source = os.path.join(temp, "source")
            output = os.path.join(temp, "output")
            os.makedirs(source)
            os.makedirs(output)
            parameters = self._portable_output(output)
            path = os.path.join(output, robotcar.MANIFEST_NAME)
            with open(path, encoding="utf-8") as handle:
                manifest = json.load(handle)
            manifest.update(
                {
                    "source_fingerprint": "whatever-it-used-to-be",
                    "source_fingerprint_schema": 3,
                    "producer_ast_sha256": "dead",
                    "content_version": 1,
                }
            )
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(manifest, handle)

            with mock.patch.object(robotcar, "_build_role") as build:
                robotcar.prepare_dataset(source, output, dict(parameters))
            build.assert_not_called()

    def test_incomplete_tree_without_a_source_fails_loudly(self):
        with tempfile.TemporaryDirectory() as output:
            parameters = self._portable_output(output, complete=False)
            with self.assertRaises(FileNotFoundError):
                robotcar.prepare_dataset(None, output, dict(parameters))

    def test_force_rebuilds_even_when_the_tree_is_complete(self):
        with tempfile.TemporaryDirectory() as temp:
            source = os.path.join(temp, "source")
            output = os.path.join(temp, "output")
            os.makedirs(source)
            os.makedirs(output)
            parameters = self._portable_output(output)
            with (
                mock.patch.object(
                    robotcar, "_build_role", return_value={"selected": 1}
                ) as build,
                mock.patch.object(robotcar, "_atomic_install"),
            ):
                robotcar.prepare_dataset(source, output, dict(parameters), force=True)
            self.assertTrue(build.called)


if __name__ == "__main__":
    unittest.main()
