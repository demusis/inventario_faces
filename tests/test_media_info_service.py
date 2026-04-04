from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import PROJECT_ROOT  # noqa: F401
from inventario_faces.infrastructure.media_info_service import MediaInfoService


class MediaInfoServiceTests(unittest.TestCase):
    def test_resolves_executable_from_configured_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            executable_path = Path(temp_dir) / "mediainfo.exe"
            executable_path.write_text("", encoding="utf-8")

            service = MediaInfoService(directory=temp_dir)
            self.assertEqual(str(executable_path.resolve()), service.executable_path)


if __name__ == "__main__":
    unittest.main()
