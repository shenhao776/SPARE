import os
import tempfile
import unittest

from tools.clean_data import clean


class CleanDataPresenceCacheTests(unittest.TestCase):
    def test_existing_clean_view_is_reused_without_source_fingerprint(self):
        with tempfile.TemporaryDirectory() as root:
            src = os.path.join(root, "source")
            out = os.path.join(root, "clean")
            recording = os.path.join(src, "recording")
            os.makedirs(recording)
            image = os.path.join(recording, "1000000000.jpg")
            with open(image, "wb") as handle:
                handle.write(b"pixels")
            with open(
                os.path.join(recording, "pose.txt"), "w", encoding="utf-8"
            ) as handle:
                handle.write("1000000000 0 0 0\n")

            self.assertEqual(clean(src, out, 500, dry_run=False), 0)
            copied = os.path.join(out, "recording", "1000000000.jpg")
            self.assertTrue(os.path.lexists(copied))

            os.unlink(image)
            self.assertEqual(clean(src, out, 500, dry_run=False), 0)
            self.assertTrue(os.path.lexists(copied))


if __name__ == "__main__":
    unittest.main()
