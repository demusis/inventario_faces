from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import PROJECT_ROOT  # noqa: F401
from inventario_faces.services.hashing_service import HashingService


class HashingServiceTests(unittest.TestCase):
    def test_sha512_matches_python_hashlib(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.bin"
            payload = b"inventario-faces"
            path.write_bytes(payload)

            expected = hashlib.sha512(payload).hexdigest()
            actual = HashingService().sha512(path)

            self.assertEqual(expected, actual)


if __name__ == "__main__":
    unittest.main()
