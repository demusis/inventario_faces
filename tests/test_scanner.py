from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests._bootstrap import PROJECT_ROOT  # noqa: F401
from inventario_faces.domain.config import MediaSettings
from inventario_faces.domain.entities import MediaType
from inventario_faces.services.scanner_service import ScannerService


class ScannerServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = ScannerService(
            MediaSettings(
                image_extensions=(".jpg", ".png"),
                video_extensions=(".mp4", ".avi", ".dav"),
            )
        )

    def test_classify_recognizes_supported_extensions(self) -> None:
        self.assertEqual(self.service.classify(Path("image.jpg")), MediaType.IMAGE)
        self.assertEqual(self.service.classify(Path("movie.mp4")), MediaType.VIDEO)
        self.assertEqual(self.service.classify(Path("gravador.dav")), MediaType.VIDEO)
        self.assertEqual(self.service.classify(Path("notes.txt")), MediaType.OTHER)

    def test_scan_skips_excluded_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "evidence").mkdir()
            (root / "evidence" / "photo.jpg").write_text("x", encoding="utf-8")
            (root / "inventario_faces_output").mkdir()
            (root / "inventario_faces_output" / "generated.jpg").write_text("y", encoding="utf-8")

            files = self.service.scan(root, excluded_directories={root / "inventario_faces_output"})

            self.assertEqual(files, [root / "evidence" / "photo.jpg"])


if __name__ == "__main__":
    unittest.main()
