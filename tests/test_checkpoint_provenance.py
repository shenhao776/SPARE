import os
import tempfile
import unittest
from unittest import mock

from core import checkpoint_provenance as provenance


class CheckpointProvenanceTests(unittest.TestCase):
    def test_replacement_with_same_size_and_mtime_is_rehashed(self):
        with tempfile.TemporaryDirectory() as root:
            weight = os.path.join(root, "weight.bin")
            cache = os.path.join(root, "digests.json")
            with open(weight, "wb") as handle:
                handle.write(b"AAAA")
            with mock.patch.object(provenance, "_CACHE_FILE", cache):
                expected = provenance.verified_file_sha256(weight)
                old = os.stat(weight)
                with open(weight, "wb") as handle:
                    handle.write(b"BBBB")
                os.utime(weight, ns=(old.st_atime_ns, old.st_mtime_ns))
                with self.assertRaisesRegex(RuntimeError, "digest mismatch"):
                    provenance.verified_file_sha256(weight, expected=expected)


if __name__ == "__main__":
    unittest.main()
